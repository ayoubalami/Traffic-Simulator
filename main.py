from config import CONFIG
from simulation import Simulation
from renderer import Renderer

def main():
    simulation = Simulation(CONFIG)
    renderer = Renderer(CONFIG)
    
    while renderer.is_running():
        dt = renderer.clock.tick(60) / 1000.0
        simulation.update(dt)
        renderer.render(simulation.get_render_data())
    
    renderer.close()

if __name__ == "__main__":
    main()