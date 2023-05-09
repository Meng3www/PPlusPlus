from utils.numpy_functions import uniform_vector, make_initial_prior
from bayesian_agents.joint_rsa import RSA
from recursion_schemes.recursion_schemes import ana_greedy, ana_beam
from utils.build_vocab import Vocabulary


if __name__ == '__main__':
    urls = [
        "http://images.cocodataset.org/val2014/COCO_val2014_000000060623.jpg",
        # "https://cdn.pixabay.com/photo/2019/08/14/18/51/school-bus-4406479_1280.jpg",
        "https://cdn.pixabay.com/photo/2015/01/06/11/06/london-590114_1280.jpg",
        "https://cdn.pixabay.com/photo/2016/03/04/19/15/neoplan-1236544_1280.jpg"
    ]

    # the rationality of the S1
    rat = [3.0]
    model = ["coco"]
    number_of_images = len(urls)
    # the model starts of assuming it's equally likely any image is the intended referent
    initial_image_prior = uniform_vector(number_of_images)
    initial_rationality_prior = uniform_vector(1)
    initial_speaker_prior = uniform_vector(1)
    initial_world_prior = make_initial_prior(initial_image_prior, initial_rationality_prior, initial_speaker_prior)

    # make a character level speaker, using torch model (instead of tensorflow model)
    speaker_model = RSA(seg_type="word", model=model)
    speaker_model.initialize_speakers(model)
    # set the possible images and rationalities
    # speaker_model.speaker_prior.set_features(images=urls,tf=False,rationalities=rat)  # list: 2, 2 tensors of shape (1, 256)
    speaker_model.initial_speakers[0].set_features(images=urls, tf=False,
                                                   rationalities=rat)  # list: 2, 2 tensors of shape (1, 256)
    # generate a sentence by unfolding stepwise, from the speaker: greedy unrolling used here, not beam search: much better to use beam search generally
    literal_caption = ana_greedy(
        # literal_caption = ana_beam(
        speaker_model,
        target=0,
        depth=0,
        speaker_rationality=0,
        speaker=0,
        start_from=list(""),
        initial_world_prior=initial_world_prior
        #	, beam_width=2
    )

    pragmatic_caption = ana_greedy(
    # pragmatic_caption = ana_beam(
        speaker_model,
        target=0,
        depth=1,
        speaker_rationality=0,
        speaker=0,
        start_from=list(""),
        initial_world_prior=initial_world_prior
        # , beam_width=2
    )

    print("Literal caption:\n", literal_caption)
    print("Pragmatic caption:\n", pragmatic_caption)
