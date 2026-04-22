from PIL import Image, ImageOps
import numpy as np
from scipy.sparse.linalg import lsqr, LinearOperator
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt


img = Image.open('cat.jpeg').resize((256, 256))
X_ref = np.asarray(ImageOps.grayscale(img)).astype('float32')


def blackbox():
    s = 4
    size_filter = size = 9
    PSF = np.array(
        [[np.exp(-0.5 * ((i - 4) / s) ** 2 - 0.5 * ((j - 4) / s) ** 2) for j in range(size)] for i in range(size)])
    PSF /= np.sum(PSF)

    def dctshift(PSF, center):
        m, n = PSF.shape
        i, j = center
        l = min(i, m - i - 1, j, n - j - 1)
        PP = PSF[i - l: i + l + 1, j - l: j + l + 1]
        Z1 = np.diag(np.ones(l + 1), k=l)
        Z2 = np.diag(np.ones(l), k=l + 1)
        PP = Z1 @ PP @ Z1.T + Z1 @ PP @ Z2.T + Z2 @ PP @ Z1.T + \
             Z2 @ PP @ Z2.T
        Ps = np.zeros_like(PSF)
        Ps[0:2 * l + 1, 0:2 * l + 1] = PP
        return Ps

    dct2 = lambda a: dct(dct(a.T, norm='ortho').T, norm='ortho')
    idct2 = lambda a: idct(idct(a.T, norm='ortho').T, norm='ortho')
    Pbig = np.zeros_like(X_ref)
    Pbig[0: size_filter, 0: size_filter] = PSF
    e1 = np.zeros_like(X_ref)
    e1[0][0] = 1
    S = np.divide(dct2(dctshift(Pbig, (4, 4))), dct2(e1))
    R = lambda X, S_matrix=S: idct2(np.multiply(S_matrix, dct2(X)))
    return R


R = blackbox()
n = np.random.normal(0, 0.5, size=X_ref.shape)
X_blur = R(X_ref) + n


# The Discrete Gradient Linear Operator
# grad : R ^{256 x256 } -> R^{2 x256x256 }

def grad(X):
    G = np.zeros_like([X, X])
    G[0, :, : -1] = X[:, 1:] - X[:, : -1]  # Horizontal Direction
    G[1, : -1, :] = X[1:, :] - X[: -1, :]  # Vertical Direction
    return G


# The Adjoint of the Discrete Gradient Linear Operator , the Discrete Divergence
# grad_T : R ^{2 x256x256 } -> R ^{256 x256 }
def grad_T(Y):

    G_T = np.zeros_like(Y[0])

    G_T[:, : -1] += Y[0, :, : -1]  # Corresponds to c [0]
    G_T[: -1, :] += Y[1, : -1, :]  # Corresponds to c [1]
    G_T[:, 1:] -= Y[0, :, : -1]  # Corresponds to c [0]
    G_T[1:, :] -= Y[1, : -1, :]  # Corresponds to c [1]

    return G_T


def projection(y, lam):
    for i in range(2):

        for j in range(256):

            for k in range(256):

                if y[i, j, k] > 1 / lam:
                    y[i, j, k] = 1 / lam

                if y[i, j, k] < -1 / lam:
                    y[i, j, k] = -1 / lam

    return y

# What follows is the code for ADMM
# this function is used to minimize the argmin that yields y_{k+1}
def soft_threshold(y, alpha):
    for i in range(2):
        for j in range(256):
            for k in range(256):

                if y[i,j,k] > 1:
                    y[i,j,k] = (y[i,j,k]-1)/alpha
                elif y[i,j,k] < -1:
                    y[i, j, k] = (y[i,j,k]+1)/alpha
                else:
                    y[i, j, k] = 0
    return y


# these two functions will appear (as linear operators) when we solve the argmin that yields x_{k+1}
def Roperator(x):
    return R(np.reshape(x, (256, 256)))

def Groperator(x):
    return grad_T(grad(np.reshape(x, (256, 256))))

# we take the two functions written above and define them as linear operators

R0 = LinearOperator(shape=(256 ** 2, 256 ** 2), matvec=Roperator, rmatvec=Roperator)

G0 = LinearOperator(shape=(256 ** 2, 256 ** 2), matvec=Groperator, rmatvec=Groperator)


def ADMM(iterations, lam):

    """
    :param iterations: iterations wished to be perform
    :param lam: lambda related to the function to minimize
    :return: the deblurred image
    """

    # NOTE: we use "u" instead of lambda in order to distinguish between the lambda given in the problem

    x_succ = X_blur
    y_prev = np.zeros((2, 256, 256))
    u_prev = np.zeros((2, 256, 256))
    alpha = 0.001

    for i in range(iterations):
        rhs = lam * R(X_blur) - grad_T(u_prev) + alpha * grad_T(y_prev)
        x_succ = lsqr(lam * R0 * R0 + alpha * G0, rhs.flatten())[0]
        x_succ = np.reshape(x_succ, (256, 256))
        y_tilde = u_prev+ alpha* grad(x_succ)
        y_succ = soft_threshold(y_tilde, alpha) # we use the soft threshold in order to minimize the 1-norm (that is not differentiable)
        u_succ = u_prev + alpha * (grad(x_succ) - y_succ)

        y_prev = y_succ
        u_prev = u_succ
    return x_succ


def finalGraph(iterations,lam):

    fig, axs = plt.subplots(1, 3, figsize=(20, 10))

    fig.suptitle("Final results for the ADMM", fontsize=40)
    X_deblurred = ADMM(iterations,lam)

    fig.tight_layout()
    axs[0].imshow(X_ref, cmap='Greys_r', vmin=0, vmax=255)
    axs[0].set_title('Original picture', fontsize=20)
    axs[1].imshow(X_blur, cmap='Greys_r', vmin=0, vmax=255)
    axs[1].set_title('Blurred picture', fontsize=20)
    axs[2].imshow(X_deblurred, cmap='Greys_r', vmin=0, vmax=255)
    axs[2].set_title('Best Deblurred picture', fontsize=20)
    plt.show()


finalGraph(iterations = 8,lam = 16)