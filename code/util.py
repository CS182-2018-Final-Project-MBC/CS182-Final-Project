import numpy as np

exclusions = ("Book:", "Book Talk:", "Category:", "Category Talk:", "Wikipedia:", "Wikipedia Talk", "Help:", "Talk:", "Template:", "Template talk:", "Portal:", "File:", "Module:", "User:", "User Talk:", "Draft Talk:", "List of")

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

    topic = topic.replace('-', ' ')


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


def get_old_samples():
    return [("speech", "lacrosse"), ("mantra", "dna"), ("Parthenon", "Natural Environment"), ("Feces", "Poet"),
#             ("penguin", "sans-serif"),  #sans-serif is not in the dictionary
            ("angelina jolie", "nitrogen"),("Carrie Fisher", "Death of Adolf Hitler"),("Lacrosse", "Comedian"),
            ("Dictionary", "Atmosphere of Earth"),
            ("Broadway theatre", "Wall Street"),
            ("Life expectancy", "Graphical User Interface"),
            ("Diazepam", "Death"),
            ("Moors", "Aryan"),
            ("Michelangelo", "Horror Fiction"),
            ("Jim Henson", "Gin"),
            ("Continental Army", "Computer Multitasking"),
            ("World Health Organization", "Ecosystem"),
            ("Blood pressure", "Mathematics"),
            ("War of 1812", "Queens of the Stone Age"),
            ("Onomatopoeia", "Wiki"),
            ("Church of England", "Joan Baez"),
            ("Nuclear Power", "Canadians"),
            ("Multi-sport event", "Ku Klux Klan"),
            ("Pony Express", "Augustus"),
            ("Organization", "Parthenon"),
            ("Battleship", "Dream"),
            ("The Cosby Show", "Marine biology"),
            ("DNA replication", "Muscle car"),
            ("Mammal", "Montreal"),
            ("River", "Engine"),
            ("Louis Armstrong", "Nuclear Power"),
            ("Entertainment", "Ralph Waldo Emerson"),
            ("Bilirubin", "Architecture"),
            ("Association football", "Axis powers"),
            ("World Series", "Nuclear warfare"),
            ("Sherlock Holmes","Magnetic resonance imaging"),
            ("Waterboarding","World War II"),
            ("World Trade Organization", "Ant"),
            ("Printed circuit board", "Typhoid fever"),
            ("Statistics","Renaissance"),
            ("Radio","Personal computer"),
            ("Bette Midler","Jellyfish"),
            ("Sigmund Freud","Vacuum"),
            ("Credit card","String theory"),
            ("Radiohead","Magnetic field"),
            ("Biosphere","Nobel Prize in Physiology or Medicine"),
            ("Mick Jagger","Knife"),
            ("West Indies","Gastroesophageal reflux disease"),
            ("Wesley Snipes","Computer science"),
            ("Airline","Bavaria"),
            ("Nevada","Maltose")]

article_titles = ['Southern Europe', 'United Nations Charter', 'Climate', "Alzheimer's disease", 'Mountain', 'Fashion', 
              'Bipolar junction transistor', 'Web search engine', 'Bosnia and Herzegovina', 'Transhumanism',
              'Ecology', 'War of 1812', 'Periodic table', 'Gastroesophageal reflux disease', 'Osama bin Laden',
              'Brown Bear', 'Documentary film', 'Stock market', 'Caroline Kennedy', 'Nevada',
              'Food and Drug Administration', 'Satire', 'Saturn', 'Rhodesia', 'Yellowstone National Park',
              'Megalodon', 'Chemical reaction', 'Lactose intolerance', 'Charles Bronson', 'Profession', 
              'Cologne', 'Dimension', 'RNA', 'Dancing with the Stars', 'List of U.S. states and territories by area', 
              'Southern California', 'Celsius', 'Envy', 'Humid subtropical climate', 'James II of England', 'Jacob',
              'Pulmonary embolism', 'Romanticism', 'Vermont', 'Ethernet', 'Written language', 'Early Middle Ages',
              'Google Chrome', 'Apple Lisa', 'Latitude', 'Will Smith', 'Frank Gehry', 'Technology', 'Jules Verne',
              'Luciano Pavarotti', 'Auto racing', 'Electron', 'Matter', 'Periodic table', 'Breakfast', 'Ghetto',
              'Ashley Massaro', 'Pathology', 'Ballet', 'Bacteria', 'Persecution of Christians', 'Earth', 'Lead',
              'Jim Carrey', 'Americans', 'Coaxial cable', 'First language', 'Memory', 'Francis Ford Coppola',
              'Mel Gibson', 'Saxophone', 'Constantinople', 'Train', 'Birdman (rapper)', 'Adolf Hitler', 
              'Sociology', 'African-American music', 'Internal combustion engine', 'Eurozone', 'William Shakespeare',
              'Polynesia', 'Copyright infringement', 'Prince (musician)', 'Taxonomy (biology)', 'Apple Lisa', 
              'Metaphysics', 'Pistol', 'Euro', 'Prussia', 'Armageddon', 'Bruce Lee', 'Brain tumor', 'Sheep', 
              'Dance music', 'Embedded system', 'J. R. R. Tolkien', 'Ornithology', 'Winter War', 
              'Personal digital assistant', 'Al-Qaeda', 'Mark Wahlberg', 'Bourbon whiskey', 'Four Tops', 'Tomato',
              'Potato', 'Mathematics', 'Kidney', 'Americans', 'Chimpanzee',
              'speech','lacrosse','mantra','dna','Parthenon','Natural Environment','Feces','Poet','angelina jolie',
              'nitrogen','Carrie Fisher','Death of Adolf Hitler','Lacrosse','Comedian','Dictionary',
              'Atmosphere of Earth','Broadway theatre','Wall Street','Life expectancy','Graphical User Interface',
              'Diazepam','Death','Moors', 'Aryan','Michelangelo','Horror Fiction','Jim Henson','Gin',
              'Continental Army','Computer Multitasking','World Health Organization','Ecosystem','Blood pressure',
              'Mathematics','War of 1812','Queens of the Stone Age','Onomatopoeia','Wiki','Church of England',
              'Joan Baez','Nuclear Power','Canadians','Multi-sport event','Ku Klux Klan','Pony Express','Augustus',
              'Organization','Parthenon','Battleship','Dream','The Cosby Show','Marine biology','DNA replication',
              'Muscle car','Mammal','Montreal','River','Engine','Louis Armstrong','Nuclear Power','Entertainment',
              'Ralph Waldo Emerson','Bilirubin','Architecture','Association football', 'Axis powers','World Series',
              'Nuclear warfare','Sherlock Holmes','Magnetic resonance imaging','Waterboarding','World War II',
              'World Trade Organization','Ant','Printed circuit board','Typhoid fever','Statistics','Renaissance',
              'Radio','Personal computer','Bette Midler','Jellyfish','Sigmund Freud','Vacuum','Credit card',
              'String theory','Radiohead','Magnetic field','Biosphere','Nobel Prize in Physiology or Medicine',
              'Mick Jagger','Knife','West Indies','Gastroesophageal reflux disease','Wesley Snipes',
              'Computer science','Airline','Bavaria','Nevada','Maltose']

def get_samples(n, seed=42):
    assert n <= len(article_titles) // 2, "Not enough sample pages to return {} unique pairs.".format(n)
    np.random.seed(seed)
    words = np.random.choice(article_titles, size=n*2, replace=False)
    return [(w1, w2) for w1, w2 in zip(words[:n], words[n:])]
