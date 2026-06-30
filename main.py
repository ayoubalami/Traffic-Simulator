from config import CONFIG
import renderer
from renderer.renderer import Renderer  


if __name__ == "__main__":
    renderer = Renderer(CONFIG)
    
    while renderer.is_running():
        dt = renderer.clock.tick(60) / 1000.0  # seconds since last frame
        renderer.handle_events()
        renderer.update(dt)
        renderer.render()
    
    renderer.close()