class Metrics:
    def __init__(self):
        self.total_vehicles_spawned = 0
        self.total_vehicles_exited = 0
        self.total_wait_time = 0.0
        self.total_stops = 0
        self.vehicles_tracked = {}
    
    def register_vehicle(self, vehicle_id):
        self.vehicles_tracked[vehicle_id] = {
            "wait_time": 0.0,
            "stops": 0,
            "spawn_time": None,
            "exit_time": None
        }
    
    def update(self, vehicles, light_controller):
        for v in vehicles:
            if id(v) not in self.vehicles_tracked:
                self.register_vehicle(id(v))
                self.total_vehicles_spawned += 1
            
            if v.stopped:
                self.vehicles_tracked[id(v)]["wait_time"] += 1/60  # approx per frame
                self.vehicles_tracked[id(v)]["stops"] += 1
    
    def vehicle_exited(self, vehicle_id):
        if vehicle_id in self.vehicles_tracked:
            self.total_vehicles_exited += 1
            data = self.vehicles_tracked[vehicle_id]
            self.total_wait_time += data["wait_time"]
            self.total_stops += data["stops"]
            del self.vehicles_tracked[vehicle_id]
    
    def get_summary(self):
        avg_wait = self.total_wait_time / max(1, self.total_vehicles_exited)
        avg_stops = self.total_stops / max(1, self.total_vehicles_exited)
        throughput = self.total_vehicles_exited
        
        return {
            "throughput": throughput,
            "avg_wait_time": avg_wait,
            "avg_stops": avg_stops,
            "active_vehicles": len(self.vehicles_tracked)
        }