import numpy as np
import skimage


def preprocess_img(img, size):
    img = np.rollaxis(img, 0, 3)  # It becomes (640, 480, 3)
    img = skimage.transform.resize(img, size)
    img = skimage.color.rgb2gray(img)

    return img

# data = minerl.data.make("MineRLNavigateDense-v0", data_dir="../dataset/navigate")
#
# # Iterate through a single epoch gathering sequences of at most 32 steps
# for current_state, action, reward, next_state, done in data.sarsd_iter(num_epochs=1, max_sequence_len=32):
#     # Print the POV @ the first step of the sequence
#     print(current_state['pov'][0])
#
#     # Print the final reward pf the sequence!
#     print(reward[-1])
#
#     # Check if final (next_state) is terminal.
#     print(done[-1])
#
#     # ... do something with the data.
#     print("At the end of trajectories the length can be < max_sequence_len", len(reward))
