# Taken from https://github.com/smsharma/jax-conditional-flows/blob/main/models/bijectors.py under MIT license


# Modifications copyright 2025 Maximilian Schebek, Freie Universität Berlin
# Modified: 2025-10-03 - Adapted and extended for bgmat project. Further modifications should be documented here.

# ==============================================================================
"""Split coupling bijector."""

from typing import Any, Callable, Tuple
import haiku as hk
from functools import partial
from distrax._src.bijectors import bijector as base
import jax
BijectorLike = base.BijectorLike
BijectorT = base.BijectorT
from distrax._src.bijectors import block
from distrax._src.utils import conversion
import jax.numpy as jnp
from bgmat.models import embeddings
from bgmat.models.gnn_conditioner import _DenseBlock

from typing import Any, Tuple, Optional
from distrax._src.bijectors.chain import Chain
from bgmat.models.gnn_conditioner import _DenseBlock
from distrax._src.utils import math

Array = base.Array
BijectorParams = Any
default_w_init = hk.initializers.VarianceScaling(1.)

class ChainConditional(Chain):
    """Conditional Chain. Taken from https://github.com/smsharma/jax-conditional-flows/blob/main/models/bijectors.py"""
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, x: Array, context: Optional[Array] = None, lattice: Optional[Array] = None, box_length: Optional[Array] = None) -> Array:
        for bijector in reversed(self._bijectors):
            x = bijector.forward(x,  context=context, lattice=lattice, box_length=box_length)
        return x

    def inverse(self, y: Array, context: Optional[Array] = None, lattice: Optional[Array] = None, box_length: Optional[Array] = None) -> Array:
        for bijector in self._bijectors:
            y = bijector.inverse(y,  context=context, lattice=lattice, box_length=box_length)
        return y

    def forward_and_log_det(self, x, context: Optional[Array] = None, lattice: Optional[Array] = None, box_length: Optional[Array] = None) -> Tuple[Array, Array]:
        # print(f"Called with args: {args}, kwargs: {kwargs}")
        x, log_det = self._bijectors[-1].forward_and_log_det(x=x, context=context, lattice=lattice, box_length=box_length)
        for bijector in reversed(self._bijectors[:-1]):
            x, ld = bijector.forward_and_log_det(x=x, context=context, lattice=lattice, box_length=box_length)
            log_det += ld
        return x, log_det

    def inverse_and_log_det(self, y, context: Optional[Array] = None, lattice: Optional[Array] = None, box_length: Optional[Array] = None) -> Tuple[Array, Array]:
        y, log_det = self._bijectors[0].inverse_and_log_det(y, context=context, lattice=lattice,  box_length=box_length)
        for bijector in self._bijectors[1:]:
            y, ld = bijector.inverse_and_log_det(y, context=context, lattice=lattice, box_length=box_length)
            log_det += ld
        return y, log_det

class ConditionalSplitCoupling(base.Bijector):

  def __init__(self,
               split_index: int,
               event_ndims: int,
               conditioner: Callable[[Array], BijectorParams],
               bijector: Callable[[BijectorParams], base.BijectorLike],
               swap: bool = False,
               split_axis: int = -1):
    """Initializes a SplitCoupling bijector.

    Args:
      split_index: the index used to split the input. The input array will be
        split along the axis specified by `split_axis` into two parts. The first
        part will correspond to indices up to `split_index` (non-inclusive),
        whereas the second part will correspond to indices starting from
        `split_index` (inclusive).
      event_ndims: the number of event dimensions the bijector operates on. The
        `event_ndims_in` and `event_ndims_out` of the coupling bijector are both
        equal to `event_ndims`.
      conditioner: a function that computes the parameters of the inner bijector
        as a function of the unchanged part of the input. The output of the
        conditioner will be passed to `bijector` in order to obtain the inner
        bijector.
      bijector: a callable that returns the inner bijector that will be used to
        transform one of the two parts. The input to `bijector` is a set of
        parameters that can be used to configure the inner bijector. The
        `event_ndims_in` and `event_ndims_out` of the inner bijector must be
        equal, and less than or equal to `event_ndims`. If they are less than
        `event_ndims`, the remaining dimensions will be converted to event
        dimensions using `distrax.Block`.
      swap: by default, the part of the input up to `split_index` is the one
        that remains unchanged. If `swap` is True, then the other part remains
        unchanged and the first one is transformed instead.
      split_axis: the axis along which to split the input. Must be negative,
        that is, it must index from the end. By default, it's the last axis.
    """
    if split_index < 0:
      raise ValueError(
          f'The split index must be non-negative; got {split_index}.')
    if split_axis >= 0:
      raise ValueError(f'The split axis must be negative; got {split_axis}.')
    if event_ndims < 0:
      raise ValueError(
          f'`event_ndims` must be non-negative; got {event_ndims}.')
    if split_axis < -event_ndims:
      raise ValueError(
          f'The split axis points to an axis outside the event. With '
          f'`event_ndims == {event_ndims}`, the split axis must be between -1 '
          f'and {-event_ndims}. Got `split_axis == {split_axis}`.')
    self._split_index = split_index
    self._conditioner = conditioner
    self._bijector = bijector
    self._swap = swap
    self._split_axis = split_axis
    super().__init__(event_ndims_in=event_ndims)

  @property
  def bijector(self) -> Callable[[BijectorParams], base.BijectorLike]:
    """The callable that returns the inner bijector of `SplitCoupling`."""
    return self._bijector

  @property
  def conditioner(self) -> Callable[[Array], BijectorParams]:
    """The conditioner function."""
    return self._conditioner

  @property
  def split_index(self) -> int:
    """The index used to split the input."""
    return self._split_index

  @property
  def swap(self) -> bool:
    """The flag that determines which part of the input remains unchanged."""
    return self._swap

  @property
  def split_axis(self) -> int:
    """The axis along which to split the input."""
    return self._split_axis

  def _split(self, x: Array) -> Tuple[Array, Array]:
    x1, x2 = jnp.split(x, [self._split_index], self._split_axis)
    if self._swap:
      x1, x2 = x2, x1
    return x1, x2

  def _recombine(self, x1: Array, x2: Array) -> Array:
    if self._swap:
      x1, x2 = x2, x1
    return jnp.concatenate([x1, x2], self._split_axis)

  def _inner_bijector(self, params: BijectorParams) -> base.Bijector:
    """Returns an inner bijector for the passed params."""
    bijector = conversion.as_bijector(self._bijector(params))
    if bijector.event_ndims_in != bijector.event_ndims_out:
      raise ValueError(
          f'The inner bijector must have `event_ndims_in==event_ndims_out`. '
          f'Instead, it has `event_ndims_in=={bijector.event_ndims_in}` and '
          f'`event_ndims_out=={bijector.event_ndims_out}`.')
    extra_ndims = self.event_ndims_in - bijector.event_ndims_in
    if extra_ndims < 0:
      raise ValueError(
          f'The inner bijector can\'t have more event dimensions than the '
          f'coupling bijector. Got {bijector.event_ndims_in} for the inner '
          f'bijector and {self.event_ndims_in} for the coupling bijector.')
    elif extra_ndims > 0:
      bijector = block.Block(bijector, extra_ndims)
    return bijector

  def forward_and_log_det(self, x: Array,  context: Optional[Array] = None, lattice: Optional[Array] = None,  box_length: Optional[Array] = None) -> Tuple[Array, Array]:
    """Computes y = f(x) and log|det J(f)(x)|."""
    self._check_forward_input_shape(x)
    x1, x2 = self._split(x)
    params = self._conditioner(x1, context=context, lattice=lattice, box_length=box_length)
    y2, logdet = self._inner_bijector(params).forward_and_log_det(x2)
    return self._recombine(x1, y2), logdet

  def inverse_and_log_det(self, y: Array,  context: Optional[Array] = None, lattice: Optional[Array] = None,  box_length: Optional[Array] = None) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
    self._check_inverse_input_shape(y)
    y1, y2 = self._split(y)
    params = self._conditioner(y1, context=context, lattice=lattice, box_length=box_length)
    x2, logdet = self._inner_bijector(params).inverse_and_log_det(y2)
    return self._recombine(y1, x2), logdet

class PostiveScaleAffine(hk.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

    def _get_params(self):
        # Learnable scale and shift
        scale_affine = hk.get_parameter("scale_affine", shape=[], init=hk.initializers.Constant(jnp.log(jnp.exp(1)-1)))
        shift_affine = hk.get_parameter("shift_affine", shape=[], init=hk.initializers.Constant(1e-3))
        return scale_affine, shift_affine

    def forward_and_log_det(self, x):
        scale, shift = self._get_params()
        # return jnp.exp(scale) * x + shift, scale
        return jax.nn.softplus(scale) * x + shift, jnp.log(jax.nn.softplus(scale))

class PostiveScale(hk.Module):
    def __init__(self,name=None):
        super().__init__(name=name)

    def _get_params(self):
        # Learnable scale and shift
        scale_affine = hk.get_parameter("scale_affine", shape=[], init=hk.initializers.Constant(jnp.log(jnp.exp(1)-1)))
        return scale_affine

    def forward_and_log_det(self, x):
        scale = self._get_params()
        # return jnp.exp(scale) * x + shift, scale
        return jax.nn.softplus(scale) * x , jnp.log(jax.nn.softplus(scale))
    

    # def inverse_and_log_det(self, x):
    #     scale, shift = self._get_params()
    #     return jnp.exp(-scale) * (x + shift), -scale

class ShapeFlow(hk.Module):
    def __init__(self,
               name: Optional[str] = None):   
        super().__init__(name=name)
    
    def forward_and_log_det(self, v):
        v, log_det = PostiveScaleAffine().forward_and_log_det(v)
        return v, log_det
    
    def inverse_and_log_det(self, v):
        v, log_det = PostiveScaleAffine().inverse_and_log_det(v)
        return v, log_det

class ShapeScaleFlow(hk.Module):
    def __init__(self,
               name: Optional[str] = None):   
        super().__init__(name=name)
    
    def forward_and_log_det(self, v):
        v, log_det = PostiveScale().forward_and_log_det(v)
        return v, log_det
    
    def inverse_and_log_det(self, v):
        v, log_det = PostiveScale().inverse_and_log_det(v)
        return v, log_det
    
class ShapeConditionalFlow:

    def __init__(
        self,
        configurational_flow: ChainConditional,
        shape_flow: ShapeFlow,
        lower_encode_vol: float,
        upper_encode_vol: float,
        num_frequencies: int,
    ):

        self._configurational_flow = configurational_flow
        self._shape_flow = shape_flow

        self._encoding_fn_vol_n = partial(
            embeddings.circular,
            lower=lower_encode_vol,
            upper=upper_encode_vol,
            num_frequencies=num_frequencies,
        )

    def forward_and_log_det(self, x: Array, v: Array, lattice: Optional[Array]=None):

        n_batch, two_n, d = x.shape

        N = two_n // 2

        v_n, log_det_v = self._shape_flow.forward_and_log_det(v /  N)
        v = v_n * N
        default = hk.initializers.VarianceScaling(1.)

        v_n_encoded = self._encoding_fn_vol_n(v.reshape(-1,1,1) / N)
        v_n_encoded =  _DenseBlock(
                widening_factor=2,
                w_init=default,
                w_init_final=default,
            )(v_n_encoded)

        x, log_det_x = self._configurational_flow.forward_and_log_det(x, context=v_n_encoded, lattice=lattice)

        log_det = log_det_v + log_det_x
        return x, v, log_det

class ShapeTempPressConditionalFlow:

    def __init__(
        self,
        configurational_flow: ChainConditional,
        shape_flow: ShapeFlow,
        lower_encode_vol: float,
        upper_encode_vol: float,
        lower_encode_temp: float,
        upper_encode_temp: float,
        lower_encode_press: float,
        upper_encode_press: float,
        num_frequencies: int,
    ):

        self._configurational_flow = configurational_flow
        self._shape_flow = shape_flow

        self._encoding_fn_vol_n = partial(
            embeddings.positional_encoding,
            lower=lower_encode_vol,
            upper=upper_encode_vol,
            num_frequencies=num_frequencies,
        )
        self._encoding_fn_temp = partial(
            embeddings.positional_encoding,
            lower=lower_encode_temp,
            upper=upper_encode_temp,
            num_frequencies=num_frequencies,
        )
        self._encoding_fn_press = partial(
            embeddings.positional_encoding,
            lower=lower_encode_press,
            upper=upper_encode_press,
            num_frequencies=num_frequencies,
        )
    def forward_and_log_det(self, x: Array, v: Array, temps: Array, pressures: Array ):

        n_batch, two_n, d = x.shape

        N = two_n // 2

        v_n, log_det_v = self._shape_flow.forward_and_log_det(v /  N)
        v = v_n * N
        default = hk.initializers.VarianceScaling(1.)

        v_n_encoded = self._encoding_fn_vol_n(v.reshape(-1,1) / N).reshape(n_batch,1,-1)
        temp_encoded = self._encoding_fn_temp(temps).reshape(n_batch,1,-1)
        press_encoded = self._encoding_fn_press(pressures).reshape(n_batch,1,-1)

        context = jnp.concat((v_n_encoded, temp_encoded, press_encoded), axis=-1)
        x, log_det_x = self._configurational_flow.forward_and_log_det(x, context)

        log_det = log_det_v + log_det_x
        return x, v, log_det


class TupleConditionalFlow:

    def __init__(
        self,
        configurational_flow: ChainConditional,
        lower_encode_1: float,
        upper_encode_1: float,
        lower_encode_2: float,
        upper_encode_2: float,
        num_frequencies: int,
    ):

        self._configurational_flow = configurational_flow

        self._encoding_fn_1 =             embeddings.PositionalEncoding(
            lower=lower_encode_1,
            upper=upper_encode_1,
            num_frequencies=num_frequencies,
        )
        self._encoding_fn_2 = embeddings.PositionalEncoding(
            lower=lower_encode_2,
            upper=upper_encode_2,
            num_frequencies=num_frequencies,
        )
    def forward_and_log_det(self, x: Array, context_1: Array, context_2: Array ):

        n_batch, two_n, d = x.shape
        print(x.shape, context_1.shape, context_2.shape)
        circ_enc_1 = self._encoding_fn_1(context_1).reshape(n_batch,1,-1)
        circ_enc_2 = self._encoding_fn_2(context_2).reshape(n_batch,1,-1)
        print(circ_enc_1.shape)
        context_1_encoded = _DenseBlock(w_init=default_w_init, w_init_final=default_w_init, widening_factor=4) (circ_enc_1)
        context_2_encoded = _DenseBlock(w_init=default_w_init, w_init_final=default_w_init, widening_factor=4)(circ_enc_2)

        context = jnp.concat((context_1_encoded, context_2_encoded), axis=-1)
        x, log_det = self._configurational_flow.forward_and_log_det(x, context=context)

        return x, log_det
    
    def inverse_and_log_det(self, x: Array, context_1: Array, context_2: Array ):

        n_batch, two_n, d = x.shape
        print(x.shape, context_1.shape, context_2.shape)

        context_1_encoded = _DenseBlock(w_init=default_w_init, w_init_final=default_w_init, widening_factor=4) (self._encoding_fn_1(context_1).reshape(n_batch,1,-1))
        context_2_encoded = _DenseBlock(w_init=default_w_init, w_init_final=default_w_init, widening_factor=4)(self._encoding_fn_2(context_2).reshape(n_batch,1,-1))

        context = jnp.concat((context_1_encoded, context_2_encoded), axis=-1)
        x, log_det = self._configurational_flow.inverse_and_log_det(x, context=context)

        return x, log_det
    
class ScalarConditionalFlow:
    def __init__(self, flow: ChainConditional, lower_encode: float=None, upper_encode: float=None, num_frequencies: int=None):
      
        self._flow = flow
        if lower_encode is not None:
          self._encoding_fn =  embeddings.PositionalEncoding(
              lower=lower_encode,
              upper=upper_encode,
              num_frequencies=num_frequencies,
          )
    
    def forward_and_log_det(self, x: Array, context: Array=None, lattice: Array=None,box_length: Array=None):
        n_batch, two_n, d = x.shape
        
        if context is not None:
          context = context.reshape(-1,1)
          context_encoded = self._encoding_fn(context).reshape(n_batch,1,-1)
        else:
           context_encoded=None
        x,  log_det = self._flow.forward_and_log_det(x, context=context_encoded, lattice=lattice, box_length=box_length)

        return x,  log_det
    def inverse_and_log_det(self, x: Array, context: Array=None, lattice: Array=None,box_length: Array=None):
        n_batch, two_n, d = x.shape
        
        if context is not None:
          context = context.reshape(-1,1)
          context_encoded = self._encoding_fn(context).reshape(n_batch,1,-1)
        else:
           context_encoded=None
        x,  log_det = self._flow.inverse_and_log_det(x, context=context_encoded, lattice=lattice, box_length=box_length)

        return x,  log_det
    
class SplitFlow:
    def __init__(self, flow: ChainConditional):
      
        self._flow = flow

    def forward_and_log_det(self, x: Array, context: Array=None, lattice: Array=None, box_length: Array=None):
        n_batch, two_n, d = x.shape

        x,  log_det = self._flow.forward_and_log_det(x, context=context, lattice=lattice)

        return x,  log_det
   
    def inverse_and_log_det(self, x: Array, context: Array):
        n_batch, two_n, d = x.shape
        context = context.reshape(-1,1)
        context_encoded = self._encoding_fn(context).reshape(n_batch,1,-1)

        x,  log_det = self._flow.inverse_and_log_det(x, context=context_encoded)

        return x, log_det

class ConditionalBlock(base.Bijector):
  """A wrapper that promotes a bijector to a block bijector.

  A block bijector applies a bijector to a k-dimensional array of events, but
  considers that array of events to be a single event. In practical terms, this
  means that the log det Jacobian will be summed over its last k dimensions.

  For example, consider a scalar bijector (such as `Tanh`) that operates on
  scalar events. We may want to apply this bijector identically to a 4D array of
  shape [N, H, W, C] representing a sequence of N images. Doing so naively will
  produce a log det Jacobian of shape [N, H, W, C], because the scalar bijector
  will assume scalar events and so all 4 dimensions will be considered as batch.
  To promote the scalar bijector to a "block scalar" that operates on the 3D
  arrays can be done by `Block(bijector, ndims=3)`. Then, applying the block
  bijector will produce a log det Jacobian of shape [N] as desired.

  In general, suppose `bijector` operates on n-dimensional events. Then,
  `Block(bijector, k)` will promote `bijector` to a block bijector that
  operates on (k + n)-dimensional events, summing the log det Jacobian over its
  last k dimensions. In practice, this means that the last k batch dimensions
  will be turned into event dimensions.
  """

  def __init__(self, bijector: BijectorLike, ndims: int):
    """Initializes a Block.

    Args:
      bijector: the bijector to be promoted to a block bijector. It can be a
        distrax bijector, a TFP bijector, or a callable to be wrapped by
        `Lambda`.
      ndims: number of batch dimensions to promote to event dimensions.
    """
    if ndims < 0:
      raise ValueError(f"`ndims` must be non-negative; got {ndims}.")
    self._bijector = conversion.as_bijector(bijector)
    self._ndims = ndims
    super().__init__(
        event_ndims_in=ndims + self._bijector.event_ndims_in,
        event_ndims_out=ndims + self._bijector.event_ndims_out,
        is_constant_jacobian=self._bijector.is_constant_jacobian,
        is_constant_log_det=self._bijector.is_constant_log_det)

  @property
  def bijector(self) -> BijectorT:
    """The base bijector, without promoting to a block bijector."""
    return self._bijector

  @property
  def ndims(self) -> int:
    """The number of batch dimensions promoted to event dimensions."""
    return self._ndims

  def forward_and_log_det(self, x: Array, context: Optional[Array]=None,  lattice: Optional[Array]=None) -> Tuple[Array, Array]:
    """Computes y = f(x) and log|det J(f)(x)|."""
    self._check_forward_input_shape(x)
    y, log_det = self._bijector.forward_and_log_det(x)
    return y, math.sum_last(log_det, self._ndims)


