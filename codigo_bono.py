import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm_y
from scipy.special import yn
import os

# ==========================================
# Carpeta donde está este archivo
# ==========================================

carpeta = os.path.dirname(os.path.abspath(__file__))

print("Las imágenes se guardarán en:")
print(carpeta)

# ==========================================
# Malla angular
# ==========================================

phi_1d = np.linspace(0, 2*np.pi, 600)
theta_1d = np.linspace(0, np.pi, 300)

phi, theta = np.meshgrid(phi_1d, theta_1d)

# ==========================================
# Armónicos esféricos reales
# ==========================================

def Y_real(l, m, theta, phi):
    """
    Armónico esférico real:
      m > 0 -> sqrt(2) Re(Y_l^m)
      m = 0 -> Y_l^0
      m < 0 -> sqrt(2) Im(Y_l^|m|)
    """

    if m > 0:
        return np.sqrt(2) * np.real(
            sph_harm_y(l, m, theta, phi)
        )

    elif m == 0:
        return np.real(
            sph_harm_y(l, 0, theta, phi)
        )

    else:
        return np.sqrt(2) * np.imag(
            sph_harm_y(l, -m, theta, phi)
        )

# ==========================================
# Función para graficar
# ==========================================

def plot_mapa(ax, datos, titulo, cmap='RdBu_r'):

    im = ax.imshow(
        datos,
        extent=[-180, 180, -90, 90],
        origin='lower',
        aspect='auto',
        cmap=cmap,
        interpolation='bilinear'
    )

    ax.set_xlabel('Longitud [°]', fontsize=11)
    ax.set_ylabel('Latitud [°]', fontsize=11)
    ax.set_title(titulo, fontsize=12)

    return im

# ==========================================
# Figura 1: Armónicos individuales
# ==========================================

configs = [
    (2, 0, r'$Y_2^0$', 'Y20.png'),
    (2, 2, r'$Y_2^2$', 'Y22.png'),
    (4, 3, r'$Y_4^3$', 'Y43.png')
]

for l, m, label, nombre_archivo in configs:

    fig, ax = plt.subplots(figsize=(8,4))

    im = plot_mapa(
        ax,
        Y_real(l, m, theta, phi),
        label
    )

    plt.colorbar(
        im,
        ax=ax,
        shrink=0.85,
        label='Amplitud'
    )

    plt.tight_layout()

    ruta = os.path.join(carpeta, nombre_archivo)

    plt.savefig(
        ruta,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    print(f"Figura guardada: {ruta}")

# ==========================================
# Figura 2: Multipolos bajos
# ==========================================

terminos_bajos = [
    (2, 0, 1.0),
    (2, 2, 0.8),
    (3, 1, 0.9),
    (3, 3, 0.6),
    (4, 0, 0.7),
    (4, 2, 0.5)
]

cmb_low = sum(
    w * Y_real(l, m, theta, phi)
    for l, m, w in terminos_bajos
)

fig, ax = plt.subplots(figsize=(11,5))

im = plot_mapa(
    ax,
    cmb_low,
    r'Combinación de multipolos bajos ($\ell = 2,3,4$)'
)

plt.colorbar(
    im,
    ax=ax,
    shrink=0.85,
    label=r'$\Delta T/T$ (u.a.)'
)

ax.text(
    0.02,
    0.05,
    r'$\theta \sim 180^\circ/\ell \gtrsim 45^\circ$',
    transform=ax.transAxes,
    fontsize=9,
    bbox=dict(
        boxstyle='round',
        facecolor='white',
        alpha=0.8
    )
)

plt.tight_layout()

ruta_bajos = os.path.join(
    carpeta,
    "multipolos_bajos.png"
)

plt.savefig(
    ruta_bajos,
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print(f"Figura guardada: {ruta_bajos}")

# ==========================================
# Figura 3: Multipolos altos
# ==========================================

terminos_altos = [
    (20, 5, 1.0),
    (20,10, 0.9),
    (25, 8, 0.85),
    (25,15, 0.75),
    (30,10, 0.70),
    (30,20, 0.60)
]

cmb_high = sum(
    w * Y_real(l, m, theta, phi)
    for l, m, w in terminos_altos
)

fig, ax = plt.subplots(figsize=(11,5))

im = plot_mapa(
    ax,
    cmb_high,
    r'Combinación de multipolos altos ($\ell = 20,25,30$)'
)

plt.colorbar(
    im,
    ax=ax,
    shrink=0.85,
    label=r'$\Delta T/T$ (u.a.)'
)

ax.text(
    0.02,
    0.05,
    r'$\theta \sim 180^\circ/\ell \lesssim 9^\circ$',
    transform=ax.transAxes,
    fontsize=9,
    bbox=dict(
        boxstyle='round',
        facecolor='white',
        alpha=0.8
    )
)

plt.tight_layout()

ruta_altos = os.path.join(
    carpeta,
    "multipolos_altos.png"
)

plt.savefig(
    ruta_altos,
    dpi=300,
    bbox_inches='tight'
)

plt.close()

print(f"Figura guardada: {ruta_altos}")



x = np.linspace(0.01, 20, 2000)

# Funciones de Neumann N0, N1, N2

N0 = yn(0, x)
N1 = yn(1, x)
N2 = yn(2, x)

# Gráfica

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x, N0, label=r'$N_0(x)$', color='royalblue',   linewidth=2)
ax.plot(x, N1, label=r'$N_1(x)$', color='crimson',     linewidth=2)
ax.plot(x, N2, label=r'$N_2(x)$', color='forestgreen', linewidth=2)

# Línea de referencia en y = 0
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')

# Limitar el eje y para visualizar mejor (las divergencias en x->0 son muy grandes)
ax.set_ylim(-3, 1)

ax.set_xlabel('$x$', fontsize=13)
ax.set_ylabel('$N_n(x)$', fontsize=13)
ax.set_title('Funciones de Neumann $N_0(x)$, $N_1(x)$ y $N_2(x)$', fontsize=14)
ax.legend(fontsize=12)


plt.tight_layout()

ruta = os.path.join(carpeta, "funciones_neumann.png")
plt.savefig(ruta, dpi=200, bbox_inches='tight')
plt.close()

print(f"Figura guardada: {ruta}")
print("\nProceso finalizado correctamente.")