# CCSVD-method
Achieving full-rank equivalent updates through LoRA-family merging in generative models.  
(生成モデルへのLoRA類のマージでフルランク相当の更新を行う手法)  

### CCSVD (Clean Covariance-Optimized SVD)  

#### What is the CCSVD Method?  
CCSVD (Clean Covariance-Optimized SVD) 260915 by muooon  
This applies the update method of QPOLA optim to static weights.  

Difference Extraction and Individual Cleansing  
ΔWᵢ₍ᶜˡᵉᵃⁿ₎ = rSVDᵣ ( clamp(ΔWᵢ, μᵢ - c · σᵢ, μᵢ + c · σᵢ) )  

Whitening via Gram Matrix (Covariance)  
dᵢ = vec(ΔWᵢ₍ᶜˡᵉᵃⁿ₎) ∈ ℝᴰ,  
Gᵢⱼ = ⟨dᵢ, dⱼ⟩ = dᵢᵀ dⱼ,  
Gᵢₙᵥ = (G + εI)⁻¹,  
w = (K / (𝟏ᵀ Gᵢₙᵥ 𝟏)) Gᵢₙᵥ 𝟏  

Integration into the Base Model  
ΔW₍ᶠⁱⁿᵃˡ₎ = b × Σᵢ₌₁ᴷ (wᵢ · ΔWᵢ₍ᶜˡᵉᵃⁿ₎),  
W₍ᵐᵉʳᵍᵉᵈ₎ = W₍ᵇᵃˢᵉ₎ + ΔW₍ᶠⁱⁿᵃˡ₎  

This is applied to the integration between LoRA variants (including LyCORIS) or into models, and between models.  
Note: This is an overview; actual implementation involves further fine-tuning, etc.  
Note: During integration, the relative difference using the L2 norm is calculated to suppress variation.  

#### Why Does It Become a Pseudo-Full-Rank Update?

What we learn from QPOLA optim is that obtaining "spatial consistency" allows us to appropriately constrain the degrees of freedom of the weights.  
If we optimize this for static weights across multiple seeds, tasks, etc., using "spatial consistency", we should be able to integrate LoRAs seamlessly.
Therefore, as described above, by cleaning up noise and outliers caused by differences in seeds, tasks, etc.,  
we can synthetically obtain a flow that observes complex landscapes from multiple directions and leads to flat minima.  
This is the concept of enabling spatial characteristics close to the degrees of freedom of a full-rank update to be reproduced post-hoc.  
