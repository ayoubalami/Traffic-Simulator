class TrafficLightController:
    """Manages all traffic lights as synchronized pairs."""
    
    def __init__(self, config):
        self.config = config
        self.green_duration = 4.0
        self.yellow_duration = 2.0
        self.red_duration = 4.0
        
        self.ns_state = "green"
        self.ew_state = "red"
        self.timer = 0.0
        self.active_pair = "ns"
    
    def update(self, dt):
        self.timer += dt
        
        if self.active_pair == "ns":
            if self.ns_state == "green" and self.timer >= self.green_duration:
                self.ns_state = "yellow"
                self.timer = 0.0
            elif self.ns_state == "yellow" and self.timer >= self.yellow_duration:
                self.ns_state = "red"
                self.ew_state = "green"
                self.active_pair = "ew"
                self.timer = 0.0
        else:
            if self.ew_state == "green" and self.timer >= self.green_duration:
                self.ew_state = "yellow"
                self.timer = 0.0
            elif self.ew_state == "yellow" and self.timer >= self.yellow_duration:
                self.ew_state = "red"
                self.ns_state = "green"
                self.active_pair = "ns"
                self.timer = 0.0
    
    def get_state(self, direction):
        """Get light state for a specific direction."""
        if direction in ("north", "south"):
            return self.ns_state
        else:
            return self.ew_state

    def get_remaining_time(self):
        """Get remaining seconds for the currently active state."""
        if self.active_pair == "ns":
            current_state = self.ns_state
        else:
            current_state = self.ew_state
        
        if current_state == "green":
            return max(0.0, self.green_duration - self.timer)
        elif current_state == "yellow":
            return max(0.0, self.yellow_duration - self.timer)
        else:  # red
            return max(0.0, self.red_duration - self.timer)