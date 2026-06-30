from config import CONFIG
import renderer
from renderer.renderer import Renderer  


def main():

    renderer = Renderer(CONFIG)

    while renderer.is_running():

        renderer.handle_events()

        renderer.render()

    renderer.close()


if __name__ == "__main__":
    main()