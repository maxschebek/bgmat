import jax.numpy as jnp
import distrax

def subtract_center_of_mass(x):
    """
    Subtract the center of mass (mean) from the input tensor along the second axis.
    
    Args:
        x: Tensor of shape (n_batch, N, d).
    
    Returns:
        x_centered: Tensor with the center of mass subtracted, shape (n_batch, N, d).
    """
    com = jnp.mean(x, axis=1, keepdims=True)  # Compute the center of mass
    x_centered = x - com  # Subtract the center of mass from x
    return x_centered

def center_of_mass(x):
    """
    Computes the center of mass (mean) of the input tensor along the second axis.
    
    Args:
        x: Tensor of shape (n_batch, N, d).

    Returns:
        com: The center of mass of the input tensor, of shape (n_batch, 1, d).
    """
    com = jnp.mean(x, axis=1, keepdims=True)
    return com

def center_of_mass_norm(x):
    """
    Computes the norm of the center of mass (mean) of the input tensor along the second axis.
    
    Args:
        x: Tensor of shape (n_batch, N, d).

    Returns:
        com: The center of mass of the input tensor, of shape (n_batch,).
    """
    com = jnp.mean(x, axis=1, keepdims=False)
    com_norm = jnp.linalg.norm(com, axis=1, keepdims=False)
    return com_norm


def split_augmented(x):
    """
    Splits the input tensor along the middle dimension into two parts.

    Args:
        x: The input tensor of shape (n_batch, 2N, d).

    Returns:
        x1, x2: Two tensors of shape (n_batch, N, d), where N = 2N // 2.
    """
    n_batch, two_n, d = x.shape
    assert two_n % 2 == 0, "Middle dimension must be even"
    
    N = two_n // 2
    x1, x2 = x[:, :N, :], x[:, N:, :]
    
    return x1, x2

def concat_augmented(x1, x2):
    """
    Concatenates two tensors along the middle dimension.

    Args:
        x1, x2: Tensors of shape (n_batch, N, d).

    Returns:
        A tensor of shape (n_batch, 2N, d), obtained by concatenating x1 and x2 along the middle dimension.
    """
    n_batch, N, d = x1.shape
    assert x2.shape == (n_batch, N, d), "x1 and x2 must have the same batch size and feature dimensions"
    
    return jnp.concatenate([x1, x2], axis=1)

def swap_com(x1, x2):
    """
    Perform the center-of-mass swap operation between two arrays.
    The center of mass of x2 is added to x1, and the center of mass of x1 is subtracted from x2.
    
    Args:
        x1: Tensor of shape (n_batch, N, d).
        x2: Tensor of shape (n_batch, N, d).

    Returns:
        x1_new, x2_new: Transformed tensors with the same shape as x1 and x2.
    """
    # Compute the center of mass for x2
    com_x2 = center_of_mass(x2)

    # Swap the centers of mass between x1 and x2
    x1_new = x1 - com_x2
    x2_new = x2 - com_x2

    return x1_new, x2_new


class ShiftCenterOfMass(distrax.Bijector):
    def __init__(self, swap, event_n_dims_in):
        """
        Initializes the SwapCenterOfMass bijector.
        
        Args:
            swap: If True, the center of mass of the second part is subtracted from the first,
                  and the center of mass of the first part is added to the second part.
        """
        super().__init__(event_ndims_in=event_n_dims_in, is_constant_jacobian=True, is_constant_log_det=True)
        self._swap = swap

    def forward_and_log_det(self, x, **kwargs):
        """
        Apply the transformation to x and return both the transformed tensor and the log-det Jacobian.
        
        Args:
            x: The input tensor of shape (n_batch, 2N, d).
        
        Returns:
            A tuple (transformed_tensor, log_det_jacobian).
        """
        x1, x2 = split_augmented(x)

        # Swap the centers of mass
        if self._swap:
            x2_new, x1_new = swap_com(x2, x1)
            log_det_jacobian = -jnp.ones(x.shape[0])

        else:
            x1_new, x2_new = swap_com(x1, x2)
            log_det_jacobian = jnp.ones(x.shape[0])

        x_new = concat_augmented(x1_new,x2_new)

        return x_new, log_det_jacobian

    def inverse_and_log_det(self, x, **kwargs):
        """
        Inverse of the swap operation: apply the reverse shift (opposite of forward) 
        and return the log-det Jacobian.
        
        Args:
            y: The transformed tensor.
        
        Returns:
            A tuple (inverse_transformed_tensor, log_det_jacobian).
        """

        x1, x2 = split_augmented(x)

        # Swap the centers of mass
        if self._swap:
            x1_new, x2_new = swap_com(x1, x2)
            log_det_jacobian = jnp.ones(x.shape[0])
        else:
            x2_new, x1_new = swap_com(x2, x1)
            log_det_jacobian = -jnp.ones(x.shape[0])

        x_new = concat_augmented(x1_new,x2_new)
        return x_new, log_det_jacobian
