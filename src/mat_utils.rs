use ndarray::{s, Array, Ix2};

pub fn invert(matrix: Array<f32, Ix2>) -> Array<f32, Ix2> {
    let shape = matrix.shape();
    assert_eq!(shape[0], shape[1]);
    let n = shape[0];

    let mut augmented: Array<f32, Ix2> = Array::zeros((n, 2 * n));
    augmented.slice_mut(s![0..n, 0..n]).assign(&(matrix));
    augmented
        .slice_mut(s![0..n, n..2 * n])
        .assign(&Array::eye(n));

    for i in 0..n {
        assert_ne!(augmented[[i, i]], 0.0); // TODO handle not invertable
        for j in 0..n {
            if i != j {
                let ratio = augmented[[j, i]] / augmented[[i, i]];
                for k in 0..2 * n {
                    augmented[[j, k]] = augmented[[j, k]] - ratio * augmented[[i, k]];
                }
            }
        }
    }
    for i in 0..n {
        for j in n..2 * n {
            augmented[[i, j]] = augmented[[i, j]] / augmented[[i, i]];
        }
    }
    let mut ret: Array<f32, Ix2> = Array::zeros((n, n));
    ret.slice_mut(s![0..n, 0..n])
        .assign(&augmented.slice(s![0..n, n..2 * n]));
    ret
}

#[test]
fn test_matrix_invert() {
    use ndarray::array;
    let test: Array<f32, Ix2> = array![[1., 2., 3.], [0., 1., 4.], [5., 6., 0.]];
    let expected: Array<f32, Ix2> = array![[-24., 18., 5.], [20., -15., -4.], [-5., 4., 1.]];
    assert_eq!(invert(test), expected);
}
