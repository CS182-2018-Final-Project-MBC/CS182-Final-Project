import numpy as np

exclusions = ("Category:", "Wikipedia:", "Help:", "Talk:", "Template:", "Template talk:", "Portal:", "File:")

def valid_page(title):
    return not title.startswith(exclusions)

# Currently uses average word distances from word2vec embeddings
def get_distance(topic, model, goal, option="combined"):
    # combined, min, and average
    assert type(topic) is list and type(goal) is list
    try:
        distances = [model.distance(x, y) for x in topic for y in goal]
        if option == "combined":
            return (np.average(distances) + np.min(distances) / 2.0)  # Combination of average and minimum
        elif option == "average":
            return np.average(distances)
        elif option == "minimum":
            return np.min(distances)
        else:
            raise Exception("Invalid distance metric: {}".format(option))
    except:
        return np.Infinity


def process_word(topic, model, combine_phrases=True):
    if not topic:  #No topic. Stop.
        return []
    
    if topic in model:
        return [topic]

    # topic = topic.replace('-', ' ')


    output = []
    words = topic.split(' ')

    if combine_phrases:
        #Iteratively search word2vec for shorter and shorter phrases
        for j in range(len(words), 1, -1):
            test = '_'.join(words[:j])
            if test in model:
                return [test] + process_word(' '.join(words[j:]), model, combine_phrases)

    if words[0] in model:
        output.append(words[0])
    return output + process_word(' '.join(words[1:]), model, combine_phrases)
