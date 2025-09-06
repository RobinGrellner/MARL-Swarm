class EntityState(object):
    """physical/external base state of all entities"""
    def __init__(self):
        # physical position
        self.p_pos = None
        self.p_orientation = None
        # physical velocity
        self.p_vel = None
        # velocity in world coordinates
        self.w_vel = None

class Entity(object):
    """properties and state of physical world entity"""
    def __init__(self):
        # name
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0