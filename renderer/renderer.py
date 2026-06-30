import pygame


class Renderer:


    def __init__(self, config):

        pygame.init()

        self.config = config

        w = config["window"]["width"]
        h = config["window"]["height"]

        self.screen = pygame.display.set_mode((w, h))

        pygame.display.set_caption(config["window"]["title"])

        self.clock = pygame.time.Clock()

        self.running = True

    def is_running(self):
        return self.running

    def close(self):
        pygame.quit()

    def handle_events(self):

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                self.running = False

    def render(self):

        colors = self.config["colors"]

        self.screen.fill(colors["background"])

        self.draw_roads()

        self.draw_lane_markings()

        pygame.display.flip()

        self.clock.tick(60)

    def draw_roads(self):

        colors = self.config["colors"]

        lane_width = self.config["lane_width"]

        roads = self.config["roads"]

        w = self.config["window"]["width"]
        h = self.config["window"]["height"]

        cx = w // 2
        cy = h // 2

        # -----------------------------
        # NORTH
        # -----------------------------

        if roads["north"]["enabled"]:

            width = lane_width * (
                roads["north"]["incoming"] +
                roads["north"]["outgoing"]
            )

            pygame.draw.rect(
                self.screen,
                colors["road"],
                (
                    cx - width//2,
                    0,
                    width,
                    cy
                )
            )

        # -----------------------------
        # SOUTH
        # -----------------------------

        if roads["south"]["enabled"]:

            width = lane_width * (
                roads["south"]["incoming"] +
                roads["south"]["outgoing"]
            )

            pygame.draw.rect(
                self.screen,
                colors["road"],
                (
                    cx-width//2,
                    cy,
                    width,
                    h-cy
                )
            )

        # -----------------------------
        # WEST
        # -----------------------------

        if roads["west"]["enabled"]:

            width = lane_width * (
                roads["west"]["incoming"] +
                roads["west"]["outgoing"]
            )

            pygame.draw.rect(
                self.screen,
                colors["road"],
                (
                    0,
                    cy-width//2,
                    cx,
                    width
                )
            )

        # -----------------------------
        # EAST
        # -----------------------------

        if roads["east"]["enabled"]:

            width = lane_width * (
                roads["east"]["incoming"] +
                roads["east"]["outgoing"]
            )

            pygame.draw.rect(
                self.screen,
                colors["road"],
                (
                    cx,
                    cy-width//2,
                    w-cx,
                    width
                )
            )
            

     
    def draw_lane_markings(self):

        colors = self.config["colors"]

        w = self.config["window"]["width"]
        h = self.config["window"]["height"]

        cx = w // 2
        cy = h // 2

        lane_width = self.config["lane_width"]

        roads = self.config["roads"]

        # --------------------------------------------------
        # NORTH
        # --------------------------------------------------

        if roads["north"]["enabled"]:

            self.draw_vertical_markings(
                x=cx,
                start_y=0,
                end_y=cy,
                incoming=roads["north"]["incoming"],
                outgoing=roads["north"]["outgoing"],
                lane_width=lane_width,
                from_top=True
            )

        # --------------------------------------------------
        # SOUTH
        # --------------------------------------------------

        if roads["south"]["enabled"]:

            self.draw_vertical_markings(
                x=cx,
                start_y=cy,
                end_y=h,
                incoming=roads["south"]["incoming"],
                outgoing=roads["south"]["outgoing"],
                lane_width=lane_width,
                from_top=False
            )

        # --------------------------------------------------
        # WEST
        # --------------------------------------------------

        if roads["west"]["enabled"]:

            self.draw_horizontal_markings(
                y=cy,
                start_x=0,
                end_x=cx,
                incoming=roads["west"]["incoming"],
                outgoing=roads["west"]["outgoing"],
                lane_width=lane_width,
                from_left=True
            )

        # --------------------------------------------------
        # EAST
        # --------------------------------------------------

        if roads["east"]["enabled"]:

            self.draw_horizontal_markings(
                y=cy,
                start_x=cx,
                end_x=w,
                incoming=roads["east"]["incoming"],
                outgoing=roads["east"]["outgoing"],
                lane_width=lane_width,
                from_left=False
            )
         
         
    def draw_vertical_markings(
        self,
        x,
        start_y,
        end_y,
        incoming,
        outgoing,
        lane_width,
        from_top):

        colors = self.config["colors"]

        total = incoming + outgoing

        road_width = total * lane_width

        left = x - road_width / 2

        # Yellow center line

        center = left + outgoing * lane_width

        pygame.draw.line(
            self.screen,
            colors["yellow"],
            (center, start_y),
            (center, end_y),
            3
        )

        # White separators

        for i in range(1, total):

            if i == outgoing:
                continue

            xx = left + i * lane_width

            pygame.draw.line(
                self.screen,
                colors["white"],
                (xx, start_y),
                (xx, end_y),
                2
            )
            
            
    def draw_horizontal_markings(
        self,
        y,
        start_x,
        end_x,
        incoming,
        outgoing,
        lane_width,
        from_left):

        colors = self.config["colors"]

        total = incoming + outgoing

        road_width = total * lane_width

        top = y - road_width / 2

        center = top + outgoing * lane_width

        pygame.draw.line(
            self.screen,
            colors["yellow"],
            (start_x, center),
            (end_x, center),
            3
        )

        for i in range(1, total):

            if i == outgoing:
                continue

            yy = top + i * lane_width

            pygame.draw.line(
                self.screen,
                colors["white"],
                (start_x, yy),
                (end_x, yy),
                2
            )         