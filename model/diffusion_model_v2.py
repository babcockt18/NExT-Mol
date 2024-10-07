import math
from e3nn import o3
import torch
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius, radius_graph
from e3nn.nn import BatchNorm, NormActivation
from model.equiformer.layer_norm import EquivariantLayerNormV2
from model.equiformer.graph_attention_transformer import FeedForwardNetwork, FullyConnectedTensorProductRescale
import torch.nn as nn
from flash_attn.modules.mha import MHA
from flash_attn.modules.mlp import Mlp
from functools import partial
from model.diffusion_model_dgt import remove_mean


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor, mask=None):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)



class CoorsNorm(nn.Module):
    def __init__(self, eps=1e-8, scale_init=1.):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors):
        norm = coors.norm(dim=-1, keepdim=True)
        normed_coors = coors / norm.clamp(min=self.eps)
        return normed_coors * self.scale

class MultiCondEquiUpdate(nn.Module):
    """Update atom coordinates equivariantly, use time emb condition."""

    def __init__(self, hidden_dim, edge_dim, dist_dim, time_dim):
        super().__init__()
        self.coord_norm = CoorsNorm(scale_init=1e-2)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, hidden_dim * 2)
        )
        input_ch = hidden_dim * 2 + edge_dim + dist_dim
        update_heads = 1# + extra_heads
        self.input_lin = nn.Linear(input_ch, hidden_dim)
        self.ln = LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, update_heads, bias=False)
        )

    def forward(self, h, pos, edge_index, edge_attr, dist, time_emb):
        row, col = edge_index
        h_input = torch.cat([h[row], h[col], edge_attr, dist], dim=1)
        coord_diff = pos[row] - pos[col]
        coord_diff = self.coord_norm(coord_diff)

        if time_emb is not None:
            shift, scale = self.time_mlp(time_emb).chunk(2, dim=1)
            inv = modulate(self.ln(self.input_lin(h_input)), shift, scale) # [edge_num, hidden_size]
        else:
            inv = self.ln(self.input_lin(h_input))
        inv = torch.tanh(self.coord_mlp(inv)) # [edge_num, update_heads]

        # aggregate position
        trans = coord_diff * inv
        agg = scatter(trans, edge_index[0], 0, reduce='add', dim_size=pos.size(0))
        pos = pos + agg

        return pos


def modulate(x, shift, scale):
    # return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    return x * (1 + scale) + shift

class TPTransBlock(nn.Module):
    def __init__(self, hidden_size, attention_dropout, time_embed_dim, tp, tp_gate=False, disable_flash_attn=False, combine_norm='layernorm', gnn_proj='linear_first'):
        super(TPTransBlock, self).__init__()
        assert hidden_size % 64 == 0
        self.tp = tp
        if gnn_proj == 'linear_first':
            self.tp_proj = nn.Sequential(
                    nn.Linear(tp.tp.irreps_out.dim, hidden_size,),)

            self.tp_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.GELU(),
                nn.Linear(hidden_size * 2, hidden_size)
            )
        elif gnn_proj == 'mlp_first':
            self.tp_proj = nn.Sequential(
                    nn.Linear(tp.tp.irreps_out.dim, hidden_size * 2,),
                    nn.GELU(),
                    nn.Linear(hidden_size * 2, hidden_size))

            self.tp_mlp = nn.Linear(hidden_size, hidden_size)
        elif gnn_proj == 'mlp_mlp':
            self.tp_proj = nn.Sequential(
                    nn.Linear(tp.tp.irreps_out.dim, hidden_size * 2,),
                    nn.GELU(),
                    nn.Linear(hidden_size * 2, hidden_size))

            self.tp_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.GELU(),
                nn.Linear(hidden_size * 2, hidden_size)
            )
        else:
            raise NotImplementedError(f'gnn_proj={gnn_proj} is not implemented')
        
        self.tp_norm = LayerNorm(hidden_size, elementwise_affine=False)

        self.attn = MHA(
            embed_dim=hidden_size,
            num_heads=hidden_size // 64,
            cross_attn=False,
            dropout=attention_dropout,
            causal=False,
            fused_bias_fc=False,
            use_flash_attn=not disable_flash_attn,
            return_residual=False
        )
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=hidden_size * 4,
            activation=F.gelu,
            return_residual=False)
        self.tp_gate = tp_gate
        if self.tp_gate:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_embed_dim, 9 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_embed_dim, 6 * hidden_size, bias=True)
            )

        norm = partial(LayerNorm, elementwise_affine=False) if True else partial(nn.BatchNorm1d, affine=False)
        self.norm1 = norm(hidden_size)
        self.norm2 = norm(hidden_size)
        self.pre_norm = True
        self.flash_attn = not disable_flash_attn

    def forward(self, x, ptr, max_seqlen, time, edge_time, node_attr, edge_index, edge_attr, edge_sh, reduce='mean'):
        ## fixme: add dropout layers
        ## tensor product
        out_tp = self.tp(node_attr, edge_index, edge_attr, edge_sh, time, edge_time, reduce=reduce)

        ## DiT
        if self.pre_norm:
            if self.tp_gate:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_tp, scale_tp, gate_tp = self.adaLN_modulation(time).chunk(9, dim=1) # shape = [node_num, hidden_size]
                
                x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), cu_seqlens=ptr, max_seqlen=max_seqlen)
                x = x + gate_tp * self.tp_mlp(modulate(self.tp_norm(self.tp_proj(out_tp)), shift_tp, scale_tp)) ## this is added by me, the other part of the transfomrers follows DiT and FlashAttn
            else:
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(time).chunk(6, dim=1) # shape = [node_num, hidden_size]
                x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), cu_seqlens=ptr, max_seqlen=max_seqlen)
                x = x + self.tp_proj(out_tp) ## this is added by me, the other part of the transfomrers follows DiT and FlashAttn
            x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(time).chunk(6, dim=1)
            x = modulate(self.norm1(x + gate_msa * self.attn(x, cu_seqlens=ptr, max_seqlen=max_seqlen)), shift_msa, scale_msa)
            x = x + self.tp_proj(out_tp) ## this is added by me, the other part of the transfomrers follows DiT and FlashAttn
            x = modulate(self.norm2(x + gate_mlp * self.mlp(x)), shift_mlp, scale_mlp)
        return x, out_tp


class TensorProductConvLayerTime(torch.nn.Module):
    def __init__(self, in_irreps, sh_irreps, out_irreps, n_edge_features, time_dim, residual=True, norm='batchnorm', norm_activation=False, ffn=False, time_gate=False, pre_norm=False):
        super(TensorProductConvLayerTime, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual

        self.tp = tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

        self.fc_pre = nn.Linear(n_edge_features, n_edge_features)
        self.edge_norm = LayerNorm(n_edge_features, elementwise_affine=False)
        self.adaLN_modulation_edge = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_dim, 2 * n_edge_features, bias=True)
            )
        
        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, n_edge_features),
            nn.ReLU(),
            nn.Linear(n_edge_features, tp.weight_numel)
        )
        if norm == 'batchnorm':
            self.norm1 = BatchNorm(in_irreps) 
            self.norm2 = BatchNorm(out_irreps)
        elif norm == 'layernorm':
            self.norm1 = EquivariantLayerNormV2(in_irreps)
            self.norm2 = EquivariantLayerNormV2(out_irreps)
        elif norm == 'None' or norm == 'none':
            self.norm1 = None
            self.norm2 = None
        else:
            raise NotImplementedError(f'norm={norm} is not implemented')
        
        self.ffn1 = FullyConnectedTensorProductRescale(in_irreps, o3.Irreps('1x0e'), in_irreps, internal_weights=True, bias=False) if ffn else None
        
        self.ffn2 = FullyConnectedTensorProductRescale(out_irreps, o3.Irreps('1x0e'), out_irreps, internal_weights=True, bias=False) if ffn else None
        
        self.in_irreps = o3.Irreps(in_irreps) 
        self.out_irreps = o3.Irreps(out_irreps)
        
        self.gate_in = o3.ElementwiseTensorProduct(self.in_irreps, o3.Irreps(f'{self.in_irreps.num_irreps}x0e'))
        self.gate_out = o3.ElementwiseTensorProduct(self.out_irreps, o3.Irreps(f'{self.out_irreps.num_irreps}x0e'))
        self.time_mlp_in = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, self.in_irreps.num_irreps)
        )
        self.time_mlp_out = nn.Sequential(nn.SiLU(),
            nn.Linear(time_dim, self.out_irreps.num_irreps)
        )
        
    def forward(self, node_attr, edge_index, edge_attr, edge_sh, time=None, edge_time=None, out_nodes=None, reduce='mean'):
        node_attr_ = node_attr
        
        if self.norm1:
            node_attr_ = self.norm1(node_attr_)

        if self.ffn1:
            ones = torch.ones_like(node_attr_[:, :1])
            node_attr_ = self.ffn1(node_attr_, ones)

        node_attr_ = self.gate_in(node_attr_, self.time_mlp_in(time))

        edge_src, edge_dst = edge_index
        edge_attr = modulate(self.fc_pre(edge_attr), *self.adaLN_modulation_edge(edge_time).chunk(2, dim=1))
        tp = self.tp(node_attr_[edge_dst], edge_sh, self.fc(edge_attr))

        out_nodes = out_nodes or node_attr.shape[0]
        out = scatter(tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduce)
        
        if self.norm2:
            out = self.norm2(out)

        if self.ffn2:
            ones = torch.ones_like(out[:, :1])
            out = self.ffn2(out, ones)

        out = self.gate_out(out, self.time_mlp_out(time))

        if self.residual:
            out[:, :node_attr_.shape[-1]] += node_attr

        return out


class TensorProductConvLayer(torch.nn.Module):
    def __init__(self, in_irreps, sh_irreps, out_irreps, n_edge_features, time_dim, residual=True, norm='batchnorm', norm_activation=False, ffn=False, time_gate=False, pre_norm=False):
        super(TensorProductConvLayer, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual

        self.tp = tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, n_edge_features),
            nn.ReLU(),
            nn.Linear(n_edge_features, tp.weight_numel)
        )
        if norm == 'batchnorm':
            self.norm1 = BatchNorm(in_irreps) if pre_norm else None
            self.norm2 = BatchNorm(out_irreps) if not pre_norm else None
        elif norm == 'layernorm':
            self.norm1 = EquivariantLayerNormV2(in_irreps) if pre_norm else None
            self.norm2 = EquivariantLayerNormV2(out_irreps) if not pre_norm else None
        elif norm == 'None' or norm == 'none':
            self.norm1 = None
            self.norm2 = None
        else:
            raise NotImplementedError(f'norm={norm} is not implemented')
        
        self.ffn1 = FullyConnectedTensorProductRescale(in_irreps, o3.Irreps('1x0e'), in_irreps, internal_weights=True, bias=False) if (ffn and pre_norm) else None
        
        self.ffn2 = FullyConnectedTensorProductRescale(out_irreps, o3.Irreps('1x0e'), out_irreps, internal_weights=True, bias=False) if (ffn and not pre_norm) else None
        
        in_irreps = o3.Irreps(in_irreps) 
        if time_gate:
            self.gate = o3.ElementwiseTensorProduct(in_irreps, o3.Irreps(f'{in_irreps.num_irreps}x0e'))
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_dim, in_irreps.num_irreps)
            )
        else:
            self.gate = None
            self.time_mlp = None

    def forward(self, node_attr, edge_index, edge_attr, edge_sh, time=None, edge_time=None, out_nodes=None, reduce='mean'):
        if self.ffn1:
            ones = torch.ones_like(node_attr[:, :1])
            node_attr = self.ffn1(node_attr, ones)
        if self.norm1:
            node_attr = self.norm1(node_attr)

        if self.time_mlp:
            node_attr = self.gate(node_attr, self.time_mlp(time))

        edge_src, edge_dst = edge_index
        tp = self.tp(node_attr[edge_dst], edge_sh, self.fc(edge_attr))

        out_nodes = out_nodes or node_attr.shape[0]
        out = scatter(tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduce)
        if self.residual:
            out[:, :node_attr.shape[-1]] += node_attr
        
        if self.norm2:
            out = self.norm2(out)

        if self.ffn2:
            ones = torch.ones_like(out[:, :1])
            out = self.ffn2(out, ones)
        return out

class LearnedSinusodialposEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb
    https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = x.unsqueeze(-1)
        freqs = x * self.weights.unsqueeze(0) * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


class JODODiffusion(nn.Module):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--in_node_features', type=int, default=44)
        parser.add_argument('--in_edge_features', type=int, default=4)
        parser.add_argument('--sigma_embed_dim', type=int, default=32)
        parser.add_argument('--time_embed_dim', type=int, default=32)
        parser.add_argument('--sh_lmax', type=int, default=2)
        parser.add_argument('--ns', type=int, default=32)
        parser.add_argument('--nv', type=int, default=8)
        parser.add_argument('--hidden_size', type=int, default=512)
        parser.add_argument('--attention_dropout', type=float, default=0.0)
        parser.add_argument('--num_conv_layers', type=int, default=8)
        parser.add_argument('--max_radius', type=float, default=10)
        parser.add_argument('--radius_embed_dim', type=int, default=50)
        parser.add_argument('--scale_by_sigma', action='store_true')
        parser.add_argument('--use_second_order_repr', action='store_true', default=False)
        parser.add_argument('--norm', type=str, default='batchnorm')
        parser.add_argument('--residual', action='store_true', default=True)
        parser.add_argument('--tp_trans', action='store_true', default=True)
        parser.add_argument('--equi_update', action='store_true', default=False)
        parser.add_argument('--dit_time_step', action='store_true', default=False)
        parser.add_argument('--not_new_pos_emb', action='store_true', default=False)
        parser.add_argument('--pred_noise', action='store_true', default=True)
        parser.add_argument('--not_norm_activation', action='store_true', default=True)
        parser.add_argument('--tp_gate', action='store_true', default=False)
        parser.add_argument('--ffn', action='store_true', default=False)
        parser.add_argument('--not_time_gate', action='store_true', default=False)
        parser.add_argument('--pre_norm', action='store_true', default=False)
        parser.add_argument('--combine_norm', type=str, default='layernorm')
        parser.add_argument('--adaln_only', action='store_true', default=False)
        parser.add_argument('--gnn_proj', type=str, default='linear_first')

    def __init__(self, args):
        super(JODODiffusion, self).__init__()
        self.args = args
        self.in_node_features = args.in_node_features
        self.in_edge_features = args.in_edge_features
        # self.sigma_min = args.sigma_min
        # self.sigma_max = args.sigma_max
        self.max_radius = args.max_radius
        self.radius_embed_dim = args.radius_embed_dim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=args.sh_lmax)
        self.ns, self.nv = args.ns, args.nv
        self.scale_by_sigma = args.scale_by_sigma
        self.pred_noise = args.pred_noise
        self.adaln_only = args.adaln_only


        if self.adaln_only:
            self.node_embedding = nn.Sequential(
                nn.Linear(args.in_node_features, args.ns),
                nn.ReLU(),
                nn.Linear(args.ns, args.ns)
            )

            self.edge_embedding = nn.Sequential(
                nn.Linear(args.in_edge_features + args.radius_embed_dim, args.ns),
                nn.ReLU(),
                nn.Linear(args.ns, args.ns)
            )
        else:
            self.node_embedding = nn.Sequential(
                nn.Linear(args.in_node_features + args.time_embed_dim, args.ns),
                nn.ReLU(),
                nn.Linear(args.ns, args.ns)
            )

            self.edge_embedding = nn.Sequential(
                nn.Linear(args.in_edge_features + args.time_embed_dim + args.radius_embed_dim, args.ns),
                nn.ReLU(),
                nn.Linear(args.ns, args.ns)
            )

        self.tp_trans = self.args.tp_trans
        if self.tp_trans:
            self.new_pos_emb = not self.args.not_new_pos_emb
            if self.new_pos_emb:
                self.pos_emb_3d = LearnedSinusodialposEmb3D(64)
                self.node_embedding2 = nn.Sequential(
                    nn.Linear(args.in_node_features + self.pos_emb_3d.out_dim, args.hidden_size * 2),
                    nn.GELU(),
                    nn.Linear(args.hidden_size * 2, args.hidden_size)
                )
            else:
                self.node_embedding2 = nn.Sequential(
                    nn.Linear(args.in_node_features + 3, args.hidden_size * 2),
                    nn.ReLU(),
                    nn.Linear(args.hidden_size * 2, args.hidden_size)
                )

        self.distance_expansion = GaussianSmearing(0.0, args.max_radius, args.radius_embed_dim)
        conv_layers = []

        if args.use_second_order_repr:
            irrep_seq = [
                f'{args.ns}x0e',
                f'{args.ns}x0e + {args.nv}x1o + {args.nv}x2e',
                f'{args.ns}x0e + {args.nv}x1o + {args.nv}x2e + {args.nv}x1e + {args.nv}x2o',
                f'{args.ns}x0e + {args.nv}x1o + {args.nv}x2e + {args.nv}x1e + {args.nv}x2o + {args.ns}x0o'
            ]
        else:
            irrep_seq = [
                f'{args.ns}x0e',
                f'{args.ns}x0e + {args.nv}x1o',
                f'{args.ns}x0e + {args.nv}x1o + {args.nv}x1e',
                f'{args.ns}x0e + {args.nv}x1o + {args.nv}x1e + {args.ns}x0o'
            ]
        if self.tp_trans:
            hidden_size = args.hidden_size
            for i in range(args.num_conv_layers):
                in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
                out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
                tp_layer = TensorProductConvLayerTime if self.adaln_only else TensorProductConvLayer
                tp = tp_layer(
                    in_irreps=in_irreps,
                    sh_irreps=self.sh_irreps,
                    out_irreps=out_irreps,
                    n_edge_features=3 * args.ns,
                    time_dim=self.args.time_embed_dim,
                    residual=args.residual,
                    norm=args.norm,
                    norm_activation=not args.not_norm_activation,
                    ffn=args.ffn,
                    time_gate=not args.not_time_gate,
                    pre_norm=args.pre_norm
                )
                layer = TPTransBlock(
                    hidden_size=hidden_size,
                    attention_dropout=self.args.attention_dropout,
                    time_embed_dim=self.args.time_embed_dim,
                    tp=tp,
                    tp_gate=args.tp_gate,
                    combine_norm=args.combine_norm,
                    gnn_proj=args.gnn_proj,
                )
                conv_layers.append(layer)

            self.conv_layers = nn.ModuleList(conv_layers)
            self.final_linear = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2, bias=False),
                nn.Tanh(),
                nn.Linear(hidden_size // 2, 3, bias=False)
            )

            self.dit_time_step = self.args.dit_time_step
            if self.dit_time_step:
                self.time_mlp = TimestepEmbedder(self.args.time_embed_dim)
            else:
                learned_dim = self.args.time_embed_dim // 2
                time_dim = self.args.time_embed_dim
                sinu_pos_emb = LearnedSinusodialposEmb(learned_dim)
                self.time_mlp = nn.Sequential(
                    sinu_pos_emb,
                    nn.Linear(learned_dim + 1, time_dim),
                    nn.GELU(),
                    nn.Linear(time_dim, time_dim)
                )
        else:
            for i in range(args.num_conv_layers):
                in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
                out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
                layer = TensorProductConvLayer(
                    in_irreps=in_irreps,
                    sh_irreps=self.sh_irreps,
                    out_irreps=out_irreps,
                    n_edge_features=3 * args.ns,
                    residual=args.residual,
                    norm=args.norm,
                    norm_activation=not args.not_norm_activation,
                    ffn=args.ffn
                )
                conv_layers.append(layer)
            self.conv_layers = nn.ModuleList(conv_layers)

            self.final_linear = nn.Sequential(
                nn.Linear(args.ns * 2 + args.nv * 6, args.ns, bias=False),
                nn.Tanh(),
                nn.Linear(args.ns, 3, bias=False)
            )

            learned_dim = self.args.time_embed_dim // 2
            time_dim = self.args.time_embed_dim
            sinu_pos_emb = LearnedSinusodialposEmb(learned_dim)
            self.time_mlp = nn.Sequential(
                sinu_pos_emb,
                nn.Linear(learned_dim + 1, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim)
            )

        self.equi_update = self.args.equi_update
        if self.equi_update:
            self.update = MultiCondEquiUpdate(
                hidden_dim=self.args.hidden_size,
                edge_dim=self.args.ns,
                dist_dim=self.args.radius_embed_dim,
                time_dim=self.args.time_embed_dim,
            )

    def forward(self, data):
        node_attr, edge_index, edge_attr, edge_sh, node_noise_emb, edge_length_emb, edge_noise_emb = self.build_conv_graph(data)
        src, dst = edge_index

        if self.tp_trans:
            node_attr = self.node_embedding(node_attr)
            edge_attr = self.edge_embedding(edge_attr)
            if self.new_pos_emb:
                x = self.node_embedding2(torch.cat((data.x, self.pos_emb_3d(data.pos)), dim=1))
            else:
                x = self.node_embedding2(torch.cat((data.x, data.pos), dim=1))
            if hasattr(data, 'lm_x'):
                x += data.lm_x
            for i, layer in enumerate(self.conv_layers):
                edge_attr_ = torch.cat([edge_attr, node_attr[src, :self.ns], node_attr[dst, :self.ns]], -1)
                x, node_attr = layer(x, data.ptr.to(torch.int32), data.max_seqlen, node_noise_emb, edge_noise_emb, node_attr, edge_index, edge_attr_, edge_sh, reduce='mean')
            if self.equi_update:
                pred_pos = self.update(x, data.pos, edge_index, edge_attr, edge_length_emb, edge_noise_emb)
                pred_pos = remove_mean(pred_pos, data.batch)
                return pred_pos
            else:
                if self.pred_noise:
                    pred_noise = self.final_linear(x)
                    pred_noise = remove_mean(pred_noise, data.batch)
                    pred_pos = (data.pos - pred_noise.detach() * data.sigma_t) / data.alpha_t
                    return pred_pos, pred_noise
                else:
                    pred_pos = self.final_linear(x)
                    pred_pos = remove_mean(pred_pos, data.batch)
                    pred_noise = (data.pos - pred_pos.detach() * data.alpha_t) / data.sigma_t
                    return pred_pos, pred_noise
        else:
            node_attr = self.node_embedding(node_attr)
            edge_attr = self.edge_embedding(edge_attr)

            for layer in self.conv_layers:
                edge_attr_ = torch.cat([edge_attr, node_attr[src, :self.ns], node_attr[dst, :self.ns]], -1)
                node_attr = layer(node_attr, edge_index, edge_attr_, edge_sh, reduce='mean')

            if self.equi_update:
                pred_pos = self.update(x, data.pos, edge_index, edge_attr, edge_length_emb, edge_noise_emb)
                pred_pos = remove_mean(pred_pos, data.batch)
                return pred_pos
            else:
                pred_pos = self.final_linear(node_attr) + data.pos.detach()
                pred_pos = remove_mean(pred_pos, data.batch)
            return pred_pos

    def build_bond_conv_graph(self, data):

        bonds = data.edge_index.long()
        bond_pos = (data.pos[bonds[0]] + data.pos[bonds[1]]) / 2
        bond_batch = data.batch[bonds[0]]
        edge_index = radius(data.pos, bond_pos, self.max_radius, batch_x=data.batch, batch_y=bond_batch)

        edge_vec = data.pos[edge_index[1]] - bond_pos[edge_index[0]]
        edge_attr = self.distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = self.final_edge_embedding(edge_attr)

        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return bonds, edge_index, edge_attr, edge_sh

    def build_conv_graph(self, data):
        # this version uses noise level from jodo instead of sigma from torsional diffusion
        radius_edges = radius_graph(data.pos.float(), self.max_radius)
        edge_index = torch.cat([data.edge_index, radius_edges], 1).long()
        edge_attr = torch.cat([
            data.edge_attr,
            torch.zeros(radius_edges.shape[-1], self.in_edge_features, device=data.x.device)
        ], 0)
        node_noise_emb = self.time_mlp(data.t_cond)
        edge_noise_emb = node_noise_emb[edge_index[0].long()]
        
        if self.adaln_only:
            node_attr = data.x
            edge_attr = edge_attr
        else:
            node_attr = torch.cat([data.x, node_noise_emb], 1)
            edge_attr = torch.cat([edge_attr, edge_noise_emb], 1)

        src, dst = edge_index
        edge_vec = data.pos[dst.long()] - data.pos[src.long()]
        edge_length_emb = self.distance_expansion(edge_vec.norm(dim=-1))

        edge_attr = torch.cat([edge_attr, edge_length_emb], 1)

        edge_sh = o3.spherical_harmonics(self.sh_irreps, edge_vec, normalize=True, normalization='component')

        return node_attr, edge_index, edge_attr, edge_sh, node_noise_emb, edge_length_emb, edge_noise_emb

class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))

# Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py

def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        # t = t * 1000
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LearnedSinusodialposEmb3D(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb
    https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim)) # shape = [dim//2]
        self.out_dim = 3 * dim + 3

    def forward(self, x):
        ## x shape = [node_num, 3]
        x = x.unsqueeze(-1) # shape = [node_num, 3, 1]
        freqs = x * self.weights.unsqueeze(0).unsqueeze(0) * 2 * math.pi # shape = [node_num, 3, dim//2]
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1) # shape = [node_num, 3, dim]
        fouriered = torch.cat((x, fouriered), dim=-1) # shape = [node_num, 3, dim+1]
        fouriered = fouriered.flatten(1,2) # shape = [node_num, 3*(dim+1)]
        return fouriered
