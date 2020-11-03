from vikingzero.utils import create_minimax_lookup

state_dict_path = "/Users/befeltingu/Documents/Github/VikingZeroDev/tests/state_dict.npy"
minimax_save_path = "/Users/befeltingu/Documents/Github/VikingZeroDev/tests/minimax_state_actions.npy"

create_minimax_lookup(minimax_save_path, state_dict_path)
