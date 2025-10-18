import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd):
        if not all(k >= 0 for k in [kp, ki, kd]):
            raise ValueError("PID gains must be non-negative.")
            
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.target_position = np.zeros(3)
        self._integral = np.zeros(3)
        self._previous_error = np.zeros(3)

    def set_target(self, target_position):
        self.target_position = np.array(target_position, dtype=np.float32)
        self.clear() # Reset errors whenever a new target is set

    def update(self, current_position):

        current_position = np.array(current_position, dtype=np.float32)
        
        # --- Calculate PID terms for each axis ---
        # Proportional term
        error = self.target_position - current_position
        
        # Integral term
        self._integral += error
        
        # Derivative term
        derivative = error - self._previous_error
        
        # --- Compute the control action ---
        p_term = self.kp * error
        i_term = self.ki * self._integral
        d_term = self.kd * derivative
        
        control_action = p_term + i_term + d_term
        
        # Update the previous error for the next iteration
        self._previous_error = error
        
        # Clamp the output to be within the valid action space range [-1, 1]
        return np.clip(control_action, -1.0, 1.0)

    def clear(self):
        self._integral = np.zeros(3)
        self._previous_error = np.zeros(3)

