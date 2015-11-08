"""Problem Set 7: Particle Filter Tracking."""

import numpy as np
import cv2
from random import randint
import math

import os

# I/O directories
input_dir = "input"
output_dir = "output"
should_print = True


# Assignment code
class ParticleFilter(object):
    """A particle filter tracker, encapsulating state, initialization and update methods."""

    def __init__(self, frame, template, **kwargs):
        """Initialize particle filter object.

        Parameters
        ----------
            frame: color BGR uint8 image of initial video frame, values in [0, 255]
            template: color BGR uint8 image of patch to track, values in [0, 255]
            kwargs: keyword arguments needed by particle filter model, including:
            - num_particles: number of particles
        """
        self.num_particles = kwargs.get('num_particles', 1000)  # extract num_particles (default: 100)
        # TODO: Your code here - extract any additional keyword arguments you need and initialize state
        self.sigma = kwargs.get('sigma', 10)

        self.template = template

        #particles - x,y pairs
        self.particles = []


        #weights - same indicies as the particles (e.g. weight[3] applies to particles[3])
        #init weights to be uniform
        self.weights = np.ones(self.num_particles, dtype=np.float) / self.num_particles
        # self.weights = []

        start_near_temp = False
        buf = 30
        # x_range = np.arange(kwargs.get('x') - buf,kwargs.get('x') + kwargs.get('w') + buf).astype(np.int)
        # y_range = np.arange(kwargs.get('y') - buf,kwargs.get('y') + kwargs.get('h') + buf).astype(np.int)
        # if should_print: print 'xrange', x_range
        # if should_print: print 'yrange', y_range
        for i in range(0,self.num_particles):
            #select a random (x,y)
            frame_height = frame.shape[0]
            frame_width = frame.shape[1]
            if start_near_temp:
                self.particles.append((randint(kwargs.get('x') - buf,kwargs.get('x') + kwargs.get('w') + buf),
                                       randint(kwargs.get('y') - buf,kwargs.get('y') + kwargs.get('h') + buf)))
            else:
                self.particles.append((randint(0, frame_width), randint(0, frame_height)))


    def re_sample(self):
        # if should_print:
        #     print "sum weights", sum(self.weights)
        #     print "min", min(self.weights)
        #     print "max", max(self.weights)


        weighted_rand_particles = np.random.multinomial(self.num_particles, self.weights, size=1)[0]
        # if should_print: print "sum weighted rand parts", sum(weighted_rand_particles)
        new_particles = []
        for i in range(self.num_particles):
            for num_parts in range(weighted_rand_particles[i]):
                new_particles.append(self.particles[i])
                # if should_print: print "len new_parts", len(new_particles)

        self.particles = new_particles
        # Do we actually want to reset the weigts?
        self.weights = np.ones(self.num_particles, dtype=np.float) / self.num_particles

        # if should_print:
        #     print "len weights", len(self.weights)
        #     print "num parts", len(self.particles)
        #     print "num_particles", self.num_particles

        # weighted_rand_particles = weighted_rand_particles.astype(np.float) / sum(weighted_rand_particles)
        # if should_print:
        #     print "weighted parts", weighted_rand_particles
        #
        # self.weights = weighted_rand_particles
        # new_particles = []
        # new_weights = []
        # avg_weight = np.average(weighted_rand_particles)
        # for i in range(self.num_particles):
        #     weight = weighted_rand_particles[i]
        #     if weight >= avg_weight:
        #         new_particles.append(self.particles[i])
        #         new_weights.append(self.weights[i])
        # self.particles = new_particles
        # self.weights = np.asarray(new_weights)
        # if should_print:
        #     print "weights: ", self.weights
        # self.num_particles = len(self.particles)

    def process(self, frame):
        """Process a frame (image) of video and update filter state.

        Parameters
        ----------
            frame: color BGR uint8 image of current video frame, values in [0, 255]
        """

        # print self.particles
        # print self.num_particles
        # print self.weights

        #sample particles based on weights
        # newParticles = np.random.choice(self.num_particles, self.num_particles, True, self.weights)

        #for each particle,
        #get frame centered at that point
        #calc MSE with the template
        #add MSE to all weights by particle
        #track how much added total to normalize
        #create noise1 & noise2 - noise = np.random.normal(mu, sigma, 1)
        #add noise to x, add noise to y
        #normalize all weights by amount added

        amountAdded = 0.0
        for i in range(0, self.num_particles):
            # if should_print : print "particles", self.particles[i]
            patch = get_patch(frame, self.particles[i], self.template.shape)

            # if should_print :
            #     print "template:", self.template.shape
            #     print "frame:", frame.shape
            #     print "patch:", patch.shape

            # ignore patches at the edges of the image
            if patch.shape == self.template.shape:

                similarity = calc_similarity(self.template, patch, self.sigma)

                self.weights[i] += similarity
                amountAdded += similarity
                noise0 = np.random.normal(0, self.sigma, 1)
                noise1 = np.random.normal(0, self.sigma, 1)

                self.particles[i] = (int(self.particles[i][0] + noise0), int(self.particles[i][1] + noise1))


        if amountAdded > 0:
            self.weights /= amountAdded
            self.weights /= sum(self.weights)

        self.re_sample()

        pass  # TODO: Your code here - use the frame as a new observation (measurement) and update model

    def render(self, frame_out):
        """Visualize current particle filter state.

        Parameters
        ----------
            frame_out: copy of frame to overlay visualization on
        """
        # Note: This may not be called for all frames, so don't do any model updates here!
        # TODO: Your code here - draw particles, tracking window and a circle to indicate spread
        u_weighted_mean = 0
        v_weighted_mean = 0
        for i in range(self.num_particles):
            u = self.particles[i][0]
            v = self.particles[i][1]
            cv2.circle(frame_out, (int(u), int(v)), 1, (0, 0, 255))
            u_weighted_mean += u * self.weights[i]
            v_weighted_mean += v * self.weights[i]

        sum_dist = 0
        for i in range(self.num_particles):
            part_pt = self.particles[i]
            sum_dist += math.sqrt((part_pt[0] - u_weighted_mean)**2 + (part_pt[1] - v_weighted_mean)**2)
        radius = int(sum_dist / self.num_particles)
        center = (int(u_weighted_mean), int(v_weighted_mean))
        x, y, h, w = get_rect(center, self.template.shape)
        cv2.rectangle(frame_out, (x, y), (x + w, y + h), (0, 255, 0))
        cv2.circle(frame_out, center, radius, (0, 255, 0))


def get_patch(frame, particle, shape_needed):
    x, y, h, w = get_rect(particle, shape_needed)

    # if should_print:
    #     print "y,h,x,w", y, h, x, w

    return frame[y:y + h, x:x + w]


def get_rect(point, shape_needed):
    #upper left point (ish)
    upperLeft = point - np.array(shape_needed)/2
    x = int(upperLeft[0])
    y = int(upperLeft[1])
    h = int(shape_needed[0])
    w = int(shape_needed[1])

    return x, y, h, w

def calc_similarity(template, patch, sigma):
    mean_std_err = ((template - patch) ** 2).mean(axis=None).astype(np.float)
    similarity = math.exp(-mean_std_err / (2.0 * (sigma **2)))

    return similarity


class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker that updates its appearance model over time."""

    def __init__(self, frame, template, **kwargs):
        """Initialize appearance model particle filter object (parameters same as ParticleFilter)."""
        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)  # call base class constructor
        # TODO: Your code here - additional initialization steps, keyword arguments

    # TODO: Override process() to implement appearance model update

    # TODO: Override render() if desired (shouldn't have to, ideally)


# Driver/helper code
def get_template_rect(rect_filename):
    """Read rectangular template bounds from given file.

    The file must define 4 numbers (floating-point or integer), separated by whitespace:
    <x> <y>
    <w> <h>

    Parameters
    ----------
        rect_filename: path to file defining template rectangle

    Returns
    -------
        template_rect: dictionary specifying template bounds (x, y, w, h), as float or int

    """
    with open(rect_filename, 'r') as f:
        values = [float(v) for v in f.read().split()]
        return dict(zip(['x', 'y', 'w', 'h'], values[0:4]))


def run_particle_filter(pf_class, video_filename, template_rect, save_frames={}, **kwargs):
    """Instantiate and run a particle filter on a given video and template.

    Create an object of type pf_class, passing in initial video frame,
    template (extracted from first frame using template_rect), and any keyword arguments.

    Parameters
    ----------
        pf_class: particle filter class to instantiate (e.g. ParticleFilter)
        video_filename: path to input video file
        template_rect: dictionary specifying template bounds (x, y, w, h), as float or int
        save_frames: dictionary of frames to save {<frame number>|'template': <filename>}
        kwargs: arbitrary keyword arguments passed on to particle filter class
    """
    # Open video file
    video = cv2.VideoCapture(video_filename)

    # Initialize objects
    template = None
    pf = None
    frame_num = 0
    count = 0

    fps = 60
    #capSize = gray.shape # this is the size of my source video
    size = (int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') # note the lower case
    vout = cv2.VideoWriter()
    success = vout.open('output.mov',fourcc,fps,size,True)

    # Loop over video (till last frame or Ctrl+C is pressed)
    while True:
        try:
            # Try to read a frame
            okay, frame = video.read()
            if not okay:
                print "done"
                break  # no more frames, or can't read video

            color_frame = frame.copy()
            frame = create_simple_frame(frame)

            # Extract template and initialize (one-time only)
            if template is None:
                y = int(template_rect['y'])
                x = int(template_rect['x'])
                h = int(template_rect['h'])
                w = int(template_rect['w'])

                kwargs['x'] = x
                kwargs['y'] = y
                kwargs['h'] = h
                kwargs['w'] = w

                template = frame[y:y + h, x:x + w]

                if 'template' in save_frames:
                    cv2.imwrite(save_frames['template'], template)
                    # cv2.rectangle(color_frame, (x, y), (x + w, y + h), (0, 255, 0))
                    # cv2.imwrite("output/frame.png", color_frame)
                    # exit()
                    # cv2.rectangle(color_frame, (x, y), (x + w, y + h), (0, 255, 0))
                    # cv2.circle(color_frame, (x + w/2, y + h/2), 5, (0, 255, 0))
                    # cv2.imwrite(save_frames['template'], color_frame)

                pf = pf_class(frame, template, **kwargs)

            # Process frame
            pf.process(frame)  # TODO: implement this!

            pf.render(color_frame)
            vout.write(color_frame)

            # Render and save output, if indicated
            if kwargs['show_img']:
                if (count % 10) == 0:
                    pf.render(color_frame)
                    cv2.imshow('Frame ' + str(count), color_frame)
                count += 1
            else:
                if frame_num in save_frames:
                    pf.render(color_frame)
                    cv2.imwrite(save_frames[frame_num], color_frame)


            # Update frame number
            frame_num += 1
        except KeyboardInterrupt:  # press ^C to quit
            break

def create_simple_frame(frame):
    weighted = False

    if weighted:
        # Weighted vals
        b, g, r = cv2.split(frame)
        frame = (b * 0.3) + (g * 0.58) + (r * 0.12)
    else:
        # Gray
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float)


    return frame

def one_a_to_d():
    run_particle_filter(ParticleFilter,  # particle filter model class
        os.path.join(input_dir, "pres_debate.avi"),  # input video
        get_template_rect(os.path.join(input_dir, "pres_debate.txt")),  # suggested template window (dict)
        # Note: To specify your own window, directly pass in a dict: {'x': x, 'y': y, 'w': width, 'h': height}
        {
            'template': os.path.join(output_dir, 'ps7-1-a-1.png'),
            0: os.path.join(output_dir, '0.png'),
            25: os.path.join(output_dir, '25.png'),
            50: os.path.join(output_dir, '50.png'),
            75: os.path.join(output_dir, '75.png'),
            100: os.path.join(output_dir, '100.png'),
            125: os.path.join(output_dir, '125.png'),
            150: os.path.join(output_dir, '150.png'),
            175: os.path.join(output_dir, '175.png'),
            200: os.path.join(output_dir, '200.png'),
            225: os.path.join(output_dir, '225.png'),
            250: os.path.join(output_dir, '250.png')
            # 28: os.path.join(output_dir, 'ps7-1-a-2.png'),
            # 84: os.path.join(output_dir, 'ps7-1-a-3.png'),
            # 144: os.path.join(output_dir, 'ps7-1-a-4.png')
        },  # frames to save, mapped to filenames, and 'template' if desired
        num_particles=200, sigma=20,  measurement_noise=0.1, show_img=False)  # TODO: specify other keyword args that your model expects, e.g. measurement_noise=0.2

    # 1b
    # TODO: Repeat 1a, but vary template window size and discuss trade-offs (no output images required)

    # 1c
    # TODO: Repeat 1a, but vary the sigma_MSE parameter (no output images required)
    # Note: To add a parameter, simply pass it in here as a keyword arg and extract it back in __init__()

    # 1d
    # TODO: Repeat 1a, but try to optimize (minimize) num_particles (no output images required)


def one_e():
    run_particle_filter(ParticleFilter,
        os.path.join(input_dir, "noisy_debate.avi"),
        get_template_rect(os.path.join(input_dir, "noisy_debate.txt")),
        {
            14: os.path.join(output_dir, 'ps7-1-e-1.png'),
            32: os.path.join(output_dir, 'ps7-1-e-2.png'),
            46: os.path.join(output_dir, 'ps7-1-e-3.png')
        },
        num_particles=500, sigma=15,  measurement_noise=0.1, show_img=True)  # TODO: Tune parameters so that model can continuing tracking through noise


def two_a():
    # TODO: Implement AppearanceModelPF (derived from ParticleFilter)
    # TODO: Run it on pres_debate.avi to track Romney's left hand, tweak parameters to track up to frame 140
    run_particle_filter(AppearanceModelPF,
                        os.path.join(input_dir, "pres_debate.avi"),
                        get_template_rect(os.path.join(input_dir, "hand.txt")),
                        {
                            'template': os.path.join(output_dir, 'ps7-2-a-1.png'),
                            15: os.path.join(output_dir, 'ps7-2-a-2'),
                            50: os.path.join(output_dir, 'ps7-2-a-3.png'),
                            140: os.path.join(output_dir, 'ps7-2-a-4.png')
                        },
                        num_particles=500, sigma=15, measurement_noise=0.1, show_img=True)


def two_b():
    # TODO: Run AppearanceModelPF on noisy_debate.avi, tweak parameters to track hand up to frame 140
    run_particle_filter(AppearanceModelPF,
                        os.path.join(input_dir, "pres_debate.avi"),
                        get_template_rect(os.path.join(input_dir, "hand.txt")),
                        {
                            'template': os.path.join(output_dir, 'ps7-2-b-1.png'),
                            15: os.path.join(output_dir, 'ps7-2-b-2.png'),
                            50: os.path.join(output_dir, 'ps7-2-b-3.png'),
                            140: os.path.join(output_dir, 'ps7-2-b-4.png')
                        },
                        num_particles=500, sigma=15, measurement_noise=0.1, show_img=True)

def main():
    """ Note: Comment out parts of this code as necessary"""

    """ 1a """
    one_a_to_d()

    """ 1e """
    # one_e()

    """ 2a """
    #two_a()

    """ 2b """
    # two_b()

    # EXTRA CREDIT
    # 3: Use color histogram distance instead of MSE (you can implement a derived class similar to AppearanceModelPF)
    # 4: Implement a more sophisticated model to deal with occlusions and size/perspective changes


if __name__ == "__main__":
    main()
