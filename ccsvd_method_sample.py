import torch

'''
CCSVD (Clean Covariance-Optimized SVD) 260915 by muooon

個別の特異値クレンジング(rSVD) 全体的な干渉のホワイトニング(Gram Matrix) レイヤ相対ノルム(L2norm)
簡易SVDでノイズを抑制しつつ共分散的なモデル間の関係性を最適化
Integration Merge Method：QPOLA optim から思いついたマージ手法です

この手法を、LoRA類(LyCORIS含)同士、生成モデル同士、LoRA類と生成モデル、
それぞれに用いることで seed値、データセット、などの違う低ランク更新を適切にマージすることが可能になります 
(seedやタスク違いの更新は反転などにより打ち消し合うものですが、それを可能にする空間整合性を持たせます)
'''

# CCSVDメソッドの核心部分(クレンジング、ホワイトニング、重み付け統合)サンプルコード
# Parameters:
# t_base (torch.Tensor): ベースモデルの対象レイヤテンソル
# delta_ws (list of torch.Tensor): 複数モデル/LoRAからの差分テンソルリスト [ΔW_1, ΔW_2, ...]
# scale (float): 統合強度の係数 (b)


# --- 以下、この手法の核心部分のサンプルコードです ---


def ccsvd_merge_core(t_base, delta_ws, scale=1.0):
    device = t_base.device
    K = len(delta_ws)
    t_base_gpu = t_base.float()
    
    cleaned_delta_ws = []
    for dw in delta_ws:
        dw = dw.to(device).float()
        
        # 差分の抽出と個別クレンジング (外れ値のクリッピング & rSVD低ランク近似)
        if dw.numel() > 1:
            # 外れ値クリッピング (c = 3.5)
            dw_std = dw.std()
            dw_mean = dw.mean()
            dw = torch.clamp(dw, dw_mean - 3.5 * dw_std, dw_mean + 3.5 * dw_std)
            
            # 2Dテンソルに対するエネルギーベース rSVD (累積エネルギー95%)
            if dw.ndim == 2 and dw.shape[0] > 4 and dw.shape[1] > 4:
                try:
                    max_q = min(max(4, min(dw.shape[0], dw.shape[1]) // 2), 32)
                    U, S, V = torch.svd_lowrank(dw, q=max_q, niter=2)
                    
                    energy = S ** 2
                    cumulative_energy = torch.cumsum(energy, dim=0) / (energy.sum() + 1e-8)
                    valid_indices = torch.where(cumulative_energy >= 0.95)[0]
                    r = max(2, int(valid_indices[0].item()) + 1) if len(valid_indices) > 0 else max(2, max_q // 2)
                    
                    dw = (U[:, :r] * S[:r].unsqueeze(0)) @ V[:, :r].T
                except RuntimeError:
                    pass
                    
        cleaned_delta_ws.append(dw)
        
    if K == 1:
        corrected_delta = cleaned_delta_ws[0]
    else:
        # グラム行列 (Gram Matrix) によるホワイトニング・干渉抑制
        stacked_deltas = torch.stack(cleaned_delta_ws, dim=0)
        flat_deltas = stacked_deltas.view(K, -1)
        G = torch.mm(flat_deltas, flat_deltas.t())
        
        # 相対的な正則化項 (ε) の付与と逆行列の計算
        g_mean = G.abs().mean() + 1e-8
        eps = 1e-2 * g_mean * torch.eye(K, device=device)
        
        try:
            G_inv = torch.inverse(G + eps)
        except RuntimeError:
            G_inv = torch.pinverse(G + eps)
            
        # 重みの算出 (w = (K / (𝟏ᵀ G_inv 𝟏)) G_inv 𝟏)
        weights = G_inv.sum(dim=1, keepdim=True)
        weights = weights / (weights.sum() + 1e-8) * K
        
        # 統合差分の合成
        corrected_delta = torch.sum(
            stacked_deltas * weights.view(K, *([1] * (stacked_deltas.ndim - 1))), 
            dim=0
        )
        
    # 安全弁 (ベースモデルのノルムに対する最大20%までの変化に抑制)
    base_norm = t_base_gpu.norm() + 1e-8
    delta_norm = corrected_delta.norm() + 1e-8
    max_allowable_ratio = 0.20
    if (delta_norm / base_norm) > max_allowable_ratio:
        corrected_delta = corrected_delta * (max_allowable_ratio / (delta_norm / base_norm))
        
    # ユーザー指定のスケール (b) 適用
    corrected_delta = corrected_delta * scale
    
    # ベースモデルへの統合 (W₍ᵐᵉʳᵍᵉᵈ₎ = W₍ᵇᵃˢᵉ₎ + ΔW₍ᶠⁱⁿᵃˡ₎)
    merged_tensor = t_base_gpu + corrected_delta
    return merged_tensor.to(t_base.dtype)