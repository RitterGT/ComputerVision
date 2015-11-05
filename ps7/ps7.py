"""Problem Set 7: Particle Filter Tracking."""

import numpy as np
import cv2

import os

# I/O directories
input_dir = "input"
output_dir = "output"


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
        self.num_particles = kwargs.get('num_particles', 100)  # extract num_particles (default: 100)
        # TODO: Your code here - extract any additional keyword arguments you need and initialize state
        self.sigma = kwargs.get('num_particles', 10)


        self.template = template

        #particles - x,y pairs
        self.particles = []

        #weights - same indicies as the particles (e.g. weight[3] applies to particles[3])
        self.weights = []
        for i in range(0,self.num_particles):
            #select a random (x,y)
            self.particles.append((np.random.choice(frame.shape[1], 1), np.random.choice(frame.shape[0], 1)))
            #init weights to be uniform
            self.weights.append(1/self.num_particles)


    def process(self, frame):
        """Process a frame (image) of video and update filter state.

        Parameters
        ----------
            frame: color BGR uint8 image of current video frame, values in [0, 255]
        """
        #sample particles based on weights
        self.particles = np.random.choice(self.particles, self.num_particles, True, self.weights)

        #for each particle,
            #get frame centered at that point
            #calc MSE with the template
            #add MSE to all weights by particle
            #track how much added total to normalize
            #create noise1 & noise2 - noise = np.random.normal(mu, sigma, 1)
            #add noise to x, add noise to y
        #normalize all weights by amount added

        amountAdded = 0
        for i in range(0, self.num_particles):

            patch = get_patch(frame, self.particles[i])
            MSE = calc_mse(self.template, patch)
            self.weights[i] += MSE
            amountAdded += MSE
            noise1 = np.random.normal(0, self.sigma, 1)
            noise2 = np.random.normal(0, self.sigma, 1)
            self.particles[i][0] += noise1
            self.particles[i][1] += noise2

        self.weights /= amountAdded


        pass  # TODO: Your code here - use the frame as a new observation (measurement) and update model

    def render(self, frame_out):
        """Visualize current particle filter state.

        Parameters
        ----------
            frame_out: copy of frame to overlay visualization on
        """
        # Note: This may not be called for all frames, so don't do any model updates here!
        pass  # TODO: Your code here - draw particles, tracking window and a circle to indicate spread


def get_patch(frame, particle, shape_needed):
    #upper left point (ish)
    upperLeft = particle - np.array(shape_needed)/2
    y = upperLeft[1]
    x = upperLeft[0]
    h = shape_needed[1]
    w = shape_needed[0]


    frame[y:y + h, x:x + w]

def calc_mse(template, patch):
    return ((template - patch) ** 2).mean(axis=None)

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

    # Loop over video (till last frame or Ctrl+C is presssed)
    while True:
        try:
            # Try to read a frame
            okay, frame = video.read()
            if not okay:
                print "BREAKING"
                break  # no more frames, or can't read video

            # Extract template and initialize (one-time only)
            if template is None:
                template = frame[int(template_rect['y']):int(template_rect['y'] + template_rect['h']),
                                 int(template_rect['x']):int(template_rect['x'] + template_rect['w'])]
                if 'template' in save_frames:
                    cv2.imwrite(save_frames['template'], template)
                pf = pf_class(frame, template, **kwargs)

            # Process frame
            pf.process(frame)  # TODO: implement this!

            # Render and save output, if indicated
            if frame_num in save_frames:
                frame_out = frame.copy()
                pf.render(frame_out)
                cv2.imwrite(save_frames[frame_num], frame_out)

            # Update frame number
            frame_num += 1
        except KeyboardInterrupt:  # press ^C to quit
            break


def main():
    # Note: Comment out parts of this code as necessary

    # 1a
    # TODO: Implement ParticleFilter
    run_particle_filter(ParticleFilter,  # particle filter model class
        os.path.join(input_dir, "pres_debate.avi"),  # input video
        get_template_rect(os.path.join(input_dir, "pres_debate.txt")),  # suggested template window (dict)
        # Note: To specify your own window, directly pass in a dict: {'x': x, 'y': y, 'w': width, 'h': height}
        {
            'template': os.path.join(output_dir, 'ps7-1-a-1.png'),
            28: os.path.join(output_dir, 'ps7-1-a-2.png'),
            84: os.path.join(output_dir, 'ps7-1-a-3.png'),
            144: os.path.join(output_dir, 'ps7-1-a-4.png')
        },  # frames to save, mapped to filenames, and 'template' if desired
        num_particles=50)  # TODO: specify other keyword args that your model expects, e.g. measurement_noise=0.2

    # 1b
    # TODO: Repeat 1a, but vary template window size and discuss trade-offs (no output images required)

    # 1c
    # TODO: Repeat 1a, but vary the sigma_MSE parameter (no output images required)
    # Note: To add a parameter, simply pass it in here as a keyword arg and extract it back in __init__()

    # 1d
    # TODO: Repeat 1a, but try to optimize (minimize) num_particles (no output images required)

    # 1e
    run_particle_filter(ParticleFilter,
        os.path.join(input_dir, "noisy_debate.avi"),
        get_template_rect(os.path.join(input_dir, "noisy_debate.txt")),
        {
            14: os.path.join(output_dir, 'ps7-1-e-1.png'),
            32: os.path.join(output_dir, 'ps7-1-e-2.png'),
            46: os.path.join(output_dir, 'ps7-1-e-3.png')
        },
        num_particles=50)  # TODO: Tune parameters so that model can continuing tracking through noise

    # 2a
    # TODO: Implement AppearanceModelPF (derived from ParticleFilter)
    # TODO: Run it on pres_debate.avi to track Romney's left hand, tweak parameters to track up to frame 140

    # 2b
    # TODO: Run AppearanceModelPF on noisy_debate.avi, tweak parameters to track hand up to frame 140

    # EXTRA CREDIT
    # 3: Use color histogram distance instead of MSE (you can implement a derived class similar to AppearanceModelPF)
    # 4: Implement a more sophisticated model to deal with occlusions and size/perspective changes


if __name__ == "__main__":
    main()
