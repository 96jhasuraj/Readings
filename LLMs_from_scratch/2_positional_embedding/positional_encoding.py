import numpy as np
import matplotlib.pyplot as plt

d_hidden = 4
context_length = 8
base = 10000

def get_sinusoidal(seq_len, d_model):
    pos = np.arange(seq_len)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(base, (2 * (i // 2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    
    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = np.concatenate([sines, cosines], axis=-1)
    return pos_encoding

def apply_rope(x, seq_len, d_model):
    theta = 1.0 / (base ** (np.arange(0, d_model, 2) / d_model))
    m = np.arange(seq_len)
    m_theta = np.outer(m, theta) 
    
    x1 = x[:, 0::2]
    x2 = x[:, 1::2]
    
    cos = np.cos(m_theta)
    sin = np.sin(m_theta)
    

    x_rotated = np.zeros_like(x)
    x_rotated[:, 0::2] = x1 * cos - x2 * sin
    x_rotated[:, 1::2] = x1 * sin + x2 * cos
    return x_rotated


raw_q = np.random.random((context_length, d_hidden))
raw_k = np.random.random((context_length, d_hidden))

attn =  np.matmul(raw_q, raw_k.T)

pos_enc = get_sinusoidal(context_length, d_hidden)
q_sin = raw_q + pos_enc
k_sin = raw_k + pos_enc
attn_sin = np.matmul(q_sin, k_sin.T)

q_rope = apply_rope(raw_q, context_length, d_hidden)
k_rope = apply_rope(raw_k, context_length, d_hidden)
attn_rope = np.matmul(q_rope, k_rope.T)

fig, ax = plt.subplots(1, 3, figsize=(12, 5))

im1 = ax[0].imshow(attn_sin, cmap='viridis')
ax[0].set_title("Sinusoidal Attention (Additive)")
ax[0].set_xlabel("Key Position")
ax[0].set_ylabel("Query Position")

im2 = ax[1].imshow(attn_rope, cmap='viridis')
ax[1].set_title("RoPE Attention (Rotary)")
ax[1].set_xlabel("Key Position")
ax[1].set_ylabel("Query Position")

im3 = ax[2].imshow(attn , cmap='viridis')
ax[2].set_title("attn Attention (Basic)")
ax[2].set_xlabel("Key Position")
ax[2].set_ylabel("Query Position")

plt.colorbar(im1, ax=ax[0])
plt.colorbar(im2, ax=ax[1])
plt.colorbar(im3, ax=ax[2])

plt.show()