use ndarray::{array, concatenate, s, Array, ArrayView, Axis, Ix1, Ix2};

use super::mat_utils::invert;

#[derive(Debug)]
/// Extended Kalman Filter for attitude estimation from MARG data
pub struct MargEkf {
    /// Internal state: rotation quaternion and gyroscope biases
    state: Array<f32, Ix1>,
    /// Predicted gravity and magnetic vectors
    predicted_accel_mag: Array<f32, Ix1>,
    /// A measure of the estimated accuracy of the state estimate
    predicted_estimate_covariance: Array<f32, Ix2>,
    /// Joint variability of process noise
    process_noise_covariance: Array<f32, Ix2>,
    /// Joint variability of observation noise
    observation_noise_covariance: Array<f32, Ix2>,
    /// Optimal Kalman gain
    gain: Array<f32, Ix2>,
    /// Observation model
    observation_model: Array<f32, Ix2>,
    /// Next state
    next_state: Array<f32, Ix1>,
    /// Previous state
    prev_state: Array<f32, Ix1>,
    /// Prediction for next timestamp
    next_estimate_covariance: Array<f32, Ix2>,
    /// Reference vector for acceleration
    acc_ref: Array<f32, Ix1>,
    /// Reference vector for magnetic field
    mag_ref: Array<f32, Ix1>,
}

impl MargEkf {
    pub fn new() -> Self {
        let quat_estimate = array![1., 0., 0., 0.];
        let gyro_bias_estimate = array![0., 0., 0.];
        let initial_state = concatenate![Axis(0), quat_estimate, gyro_bias_estimate];
        let predicted_accel_mag = Array::zeros(6);

        MargEkf {
            state: initial_state.clone().reversed_axes(),
            predicted_accel_mag: predicted_accel_mag.reversed_axes(),
            predicted_estimate_covariance: Array::eye(7) * 0.01,
            process_noise_covariance: Array::eye(7) * 0.001,
            observation_noise_covariance: Array::eye(6) * 0.1,
            gain: Array::zeros((7, 6)),
            observation_model: Array::zeros((6, 7)),
            next_state: initial_state.clone().reversed_axes(),
            prev_state: initial_state.reversed_axes(),
            next_estimate_covariance: Array::zeros((7, 7)),
            acc_ref: array![0., 0., -1.].reversed_axes(),
            mag_ref: array![0., -1., 0.].reversed_axes(),
        }
    }

    pub fn predict(&mut self, w_x: f32, w_y: f32, w_z: f32, d_t: f32) {
        let q = self.state.slice(s![0..4]);
        let rotation = array![
            [-q[1], -q[2], -q[3]],
            [q[0], -q[3], q[2]],
            [q[3], q[0], -q[1]],
            [-q[2], q[1], q[0]]
        ];
        let advance = d_t / 2.0 * rotation;
        let mut transition_model: Array<f32, Ix2> = Array::eye(7);
        transition_model
            .slice_mut(s![0..4, 4..7])
            .assign(&(-advance.clone()));
        let mut control_model: Array<f32, Ix2> = Array::zeros((7, 3));
        control_model.slice_mut(s![0..4, 0..3]).assign(&advance);

        self.next_state = transition_model.dot(&self.state)
            + control_model.dot(&array![w_x, w_y, w_z].reversed_axes());

        self.next_estimate_covariance = (transition_model.dot(&self.predicted_estimate_covariance))
            .dot(&transition_model.reversed_axes())
            + self.process_noise_covariance.clone();

        let quat_mag = self.next_state.slice(s![0..4]).map(|q| q * q).sum().sqrt();
        let normalized_quat = self.next_state.slice(s![0..4]).map(|q| q / quat_mag);
        self.next_state.slice_mut(s![0..4]).assign(&normalized_quat);

        self.prev_state = self.state.clone();

        let mut observation_model: Array<f32, Ix2> = Array::zeros((6, 7));
        observation_model
            .slice_mut(s![0..3, 0..4])
            .assign(&jacobian(
                &self.prev_state.slice(s![0..4]).into(),
                &self.acc_ref,
            ));
        observation_model
            .slice_mut(s![3..6, 0..4])
            .assign(&jacobian(
                &self.prev_state.slice(s![0..4]).into(),
                &self.mag_ref,
            ));
        self.observation_model = observation_model;

        let mut predicted_accel_mag: Array<f32, Ix1> = Array::zeros(6);
        predicted_accel_mag.slice_mut(s![0..3]).assign(
            &rotation_matrix(&self.next_state)
                .reversed_axes()
                .dot(&self.acc_ref),
        );
        predicted_accel_mag.slice_mut(s![3..6]).assign(
            &rotation_matrix(&self.next_state)
                .reversed_axes()
                .dot(&self.mag_ref),
        );
        self.predicted_accel_mag = predicted_accel_mag;
    }

    fn mag_from_raw(&self, raw: Array<f32, Ix1>) -> Array<f32, Ix1> {
        // TODO: those two constants should be part of something else than this crate
        let a_inv: Array<f32, Ix2> = array![
            [2e-03, -1e-04, -1e-06],
            [-1e-04, 1e-03, 1e-05],
            [-1e-06, 1e-05, 1e-03]
        ];
        let b: Array<f32, Ix1> = array![80., 37., 105.].reversed_axes();

        let gauss_raw = a_inv.dot(&(raw.reversed_axes() - b));
        let mut gauss_n = rotation_matrix(&self.state).dot(&gauss_raw);
        gauss_n[2] = 0.;
        let norm = (gauss_n[0].powi(2) + gauss_n[1].powi(2)).sqrt();
        gauss_n = gauss_n / norm;
        rotation_matrix(&self.state).reversed_axes().dot(&gauss_n)
    }

    pub fn update(&mut self, a: [f32; 3], m: [f32; 3]) {
        self.gain = self
            .next_estimate_covariance
            .dot(&self.observation_model.clone().reversed_axes())
            .dot(&invert(
                self.observation_model
                    .dot(&self.next_estimate_covariance)
                    .dot(&self.observation_model.clone().reversed_axes())
                    + self.observation_noise_covariance.clone(),
            ));
        let magnetic = self.mag_from_raw(array![m[0], m[1], m[2]]);
        let mut accel = (array![a[0], a[1], a[2]]).reversed_axes();
        let norm = (accel[0].powi(2) + accel[1].powi(2) + accel[2].powi(2)).sqrt();
        accel = accel / norm;
        let measurement = concatenate![Axis(0), accel, magnetic];
        self.state = self.next_state.clone()
            + self
                .gain
                .dot(&(measurement - self.predicted_accel_mag.clone()));
        let quat_mag = self.state.slice(s![0..4]).map(|q| q * q).sum().sqrt();
        let normalized_quat = self.state.slice(s![0..4]).map(|q| q / quat_mag);
        self.state.slice_mut(s![0..4]).assign(&normalized_quat);
        self.predicted_estimate_covariance = (Array::eye(7)
            - self.gain.dot(&self.observation_model))
        .dot(&self.next_estimate_covariance)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ekf_initial_predict() {
        let mut ekf = MargEkf::new();
        ekf.predict(-0.01, 0.02, 0.03, 0.1);
        assert_eq!(
            ekf.next_state,
            array![
                0.99999833,
                -0.0004999992,
                0.0009999984,
                0.0014999975,
                0.0,
                0.0,
                0.0
            ]
        )
    }

    #[test]
    fn test_ekf_initial_update() {
        let mut ekf = MargEkf::new();
        ekf.predict(-0.01, 0.02, 0.03, 0.1);
        ekf.update(
            [-0.038308594, -0.22745728, 10.362474],
            [342.98535, 165.42484, 331.73993],
        );
        assert_eq!(
            ekf.gain,
            array![
                [0., 0., -0.117021285, 0., -0.11702128, 0.],
                [0., -0.11716259, 0., 0., 0., 0.117162585],
                [0.15301873, 0., 0., 0., 0., 0.],
                [0., 0., 0., -0.15301873, 0., 0.],
                [0., 0.005313497, 0., 0., 0., -0.005313496],
                [-0.006939625, 0., 0., 0., 0., 0.],
                [0., 0., 0., 0.006939625, 0., 0.]
            ]
        );
        assert_eq!(
            ekf.state,
            array![
                0.97281927,
                0.0035848243,
                0.00019932055,
                -0.23153809,
                -0.00012722983,
                0.000039538067,
                0.006821011
            ]
        );
        assert_eq!(
            ekf.predicted_estimate_covariance,
            array![
                [0.005851064, 0., 0., 0., 0., 0., 0.],
                [0., 0.0058581303, 0., 0., -0.00026567484, 0., 0.],
                [0., 0., 0.0076509374, 0., 0., -0.0003469813, 0.],
                [0., 0., 0., 0.0076509374, 0., 0., -0.0003469813],
                [0., -0.0002656748, 0., 0., 0.010989373, 0., 0.],
                [0., 0., -0.00034698128, 0., 0., 0.010993061, 0.],
                [0., 0., 0., -0.00034698128, 0., 0., 0.010993061]
            ]
        );
    }

    #[test]
    fn test_ekf_initial_state() {
        let ekf = MargEkf::new();
        println!("{:?}", ekf);
        assert_eq!(ekf.state, array![1., 0., 0., 0., 0., 0., 0.]);
    }
}

fn rotation_matrix(quat: &Array<f32, Ix1>) -> Array<f32, Ix2> {
    let e00 = quat[0].powi(2) + quat[1].powi(2) - quat[2].powi(2) - quat[3].powi(2);
    let e01 = 2.0 * (quat[1] * quat[2] - quat[0] * quat[3]);
    let e02 = 2.0 * (quat[1] * quat[3] + quat[0] * quat[2]);
    let e10 = 2.0 * (quat[1] * quat[2] + quat[0] * quat[3]);
    let e11 = quat[0].powi(2) - quat[1].powi(2) + quat[2].powi(2) - quat[3].powi(2);
    let e12 = 2.0 * (quat[2] * quat[3] - quat[0] * quat[1]);
    let e20 = 2.0 * (quat[1] * quat[3] - quat[0] * quat[2]);
    let e21 = 2.0 * (quat[2] * quat[3] + quat[0] * quat[1]);
    let e22 = quat[0].powi(2) - quat[1].powi(2) - quat[2].powi(2) + quat[3].powi(2);
    array![[e00, e01, e02], [e10, e11, e12], [e20, e21, e22]]
}

fn jacobian(quat: &ArrayView<f32, Ix1>, reference: &Array<f32, Ix1>) -> Array<f32, Ix2> {
    let e00 = quat[0] * reference[0] + quat[3] * reference[1] - quat[2] * reference[2];
    let e01 = quat[1] * reference[0] + quat[2] * reference[1] + quat[3] * reference[2];
    let e02 = -quat[2] * reference[0] + quat[1] * reference[1] - quat[0] * reference[2];
    let e03 = -quat[3] * reference[0] + quat[0] * reference[1] + quat[1] * reference[2];
    let e10 = -quat[3] * reference[0] + quat[0] * reference[1] + quat[1] * reference[2];
    let e11 = quat[2] * reference[0] - quat[1] * reference[1] + quat[0] * reference[2];
    let e12 = quat[1] * reference[0] + quat[2] * reference[1] + quat[3] * reference[2];
    let e13 = -quat[0] * reference[0] - quat[3] * reference[1] + quat[2] * reference[2];
    let e20 = quat[2] * reference[0] - quat[1] * reference[1] + quat[0] * reference[2];
    let e21 = quat[3] * reference[0] - quat[0] * reference[1] - quat[1] * reference[2];
    let e22 = quat[0] * reference[0] + quat[3] * reference[1] - quat[2] * reference[2];
    let e23 = quat[1] * reference[0] + quat[2] * reference[1] + quat[3] * reference[2];

    2.0 * array![
        [e00, e01, e02, e03],
        [e10, e11, e12, e13],
        [e20, e21, e22, e23]
    ]
}
