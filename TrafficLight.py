
class TrafficLight:
    def __init__(self):
        self.state = "green"   # "green", "yellow", "red"
        self.timer = 0

    def update(self, dt):
        self.timer += dt
        # Simple cycle: green 3s -> yellow 1s -> red 3s
        if self.state == "green" and self.timer >= 3.0:
            self.state = "yellow"
            self.timer = 0
        elif self.state == "yellow" and self.timer >= 1.0:
            self.state = "red"
            self.timer = 0
        elif self.state == "red" and self.timer >= 3.0:
            self.state = "green"
            self.timer = 0

    def get_color(self, colors_config):
        return colors_config[self.state]

