"""Test the ``dynml.ml.dimred`` module.

This module tests the ``dynml.ml.dimred`` module.
"""


# import built-in python-package code
# None
# import external python-package code
from torch import float64, randn, set_default_dtype, tensor
from torch.cuda import is_available
# import internal python-package code
from dynml.ml.paramfunc import Affine
from dynml.ml.dimred import local_pca, DimRed, DimRedComposition


# test local_pod()
def test_local_pod() -> None:
    """Test the ``local_pca()`` method.

    This method tests the ``local_pca()`` method. First, we sample two
    points close to the origin on the x-axis and sample two points far away
    from the origin on the y-axis. Finally, we apply the ``local_pca()`` method
    to the data set and check if projecting onto the dominant singular vector
    reconstructs the data set on the x-axis.

    | **Args**
    |   None

    | **Returns**
    |   None

    | **Raises**
    |   None

    | **References**
    |   None
    """
    # sample data set
    X = tensor([[0.5, 0.0], [-0.5, 0.0], [0.0, 10.0], [0.0, -10.0]])
    # test local_pod()
    center = tensor([0.0, 0.0])
    r = 1.0
    Phi_T, _ = local_pca(X, center, r)
    # project data set onto dominant singular vector
    X_proj = X[0:2] @ Phi_T[0:1].T @ Phi_T[0:1]
    assert X_proj.allclose(X[0:2])


# test DimRed
def test_DimRed() -> None:
    """Test the ``DimRed`` class.

    This method tests the ``DimRed`` class. In particular, we instantiate the
    class and test ``dims_in``, ``dims_out``, ``enc``, ``dec``, ``dim_latent``,
    and ``forward()``. We also test ``torch.nn.Module`` backwards
    differentiation with respect to ``forward()``.

    | **Args**
    |   None

    | **Returns**
    |   None

    | **Raises**
    |   None

    | **References**
    |   None
    """
    # set torch to float64
    set_default_dtype(float64)
    # find device
    device = 'cuda' if is_available() else 'cpu'
    # instantiate DimRed
    enc = Affine(2, 1, randn((1, 1)), randn((2, 1)))
    dec = Affine(1, 2, randn((1, 2)), randn((1, 2)))
    dimred = DimRed(enc, dec).to(device)
    # test dims_in
    assert dimred.dims_in == (2,)
    # test dims_out
    assert dimred.dims_out == (2,)
    # test enc
    assert dimred.enc == enc
    # test dec
    assert dimred.dec == dec
    # test dims_latent
    assert dimred.dims_latent == (1,)
    # test forward()
    x = tensor([[1.0, 1.0], [2.0, 2.0]], device=device)
    output = dimred(x)
    desired = dec(enc(x))
    assert output.allclose(desired)
    # test torch.nn.Module backwards differentiation
    x = tensor([[1.0, 1.0], [2.0, 2.0]], device=device, requires_grad=True)
    y = dimred(x)
    y.sum().backward()


# test DimRedComposition
def test_DimRedComposition() -> None:
    """Test the ``DimRedComposition`` class.

    This method tests the ``DimRedComposition`` class. In particular, we
    instantiate the class and test ``dims_in``, ``dims_out``, ``enc``, ``dec``,
    ``dim_latent``, and ``forward()``. We also test ``torch.nn.Module``
    backwards differentiation with respect to ``forward()``.

    | **Args**
    |   None

    | **Returns**
    |   None

    | **Raises**
    |   None

    | **References**
    |   None
    """
    # set torch to float64
    set_default_dtype(float64)
    # find device
    device = 'cuda' if is_available() else 'cpu'
    # instantiate DimRedComposition
    enc1 = Affine(3, 2, randn((1, 2)), randn((3, 2)))
    dec1 = Affine(2, 3, randn((1, 3)), randn((2, 3)))
    enc2 = Affine(2, 1, randn((1, 1)), randn((2, 1)))
    dec2 = Affine(1, 2, randn((1, 2)), randn((1, 2)))
    dimred1 = DimRed(enc1, dec1)
    dimred2 = DimRed(enc2, dec2)
    dimred = DimRedComposition((dimred1, dimred2)).to(device)
    # test dims_in
    assert dimred.dims_in == (3,)
    # test dims_out
    assert dimred.dims_out == (3,)
    # test enc
    x = tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], device=device)
    output = dimred.enc(x)
    desired = enc2(enc1(x))
    assert output.allclose(desired)
    # test dec
    x = tensor([[1.0], [2.0]], device=device)
    output = dimred.dec(x)
    desired = dec1(dec2(x))
    assert output.allclose(desired)
    # test dims_latent
    assert dimred.dims_latent == (1,)
    # test forward()
    x = tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], device=device)
    output = dimred(x)
    desired = dec1(dec2(enc2(enc1(x))))
    assert output.allclose(desired)
    # test torch.nn.Module backwards differentiation
    x = tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], device=device,
               requires_grad=True)
    desired = dimred(x)
    desired.sum().backward()
