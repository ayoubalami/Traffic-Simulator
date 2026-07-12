from config import CONFIG, build_runtime_config
from simulation import Simulation
from renderer import Renderer

def main():
    runtime_config = build_runtime_config(CONFIG)
    simulation = Simulation(runtime_config)
    renderer = Renderer(runtime_config)
    
    while renderer.is_running():
        # Cap and measure the frame once.  Measuring it again in Renderer.render
        # made the simulation advance only about half as fast as real time.
        dt = renderer.clock.tick(60) / 1000.0
        dt *= runtime_config["simulation"].get("time_scale", 1.0)
        simulation.update(dt)
        renderer.render(simulation.get_render_data())
    
    renderer.close()

if __name__ == "__main__":
    main()
