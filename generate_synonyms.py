# -*- coding: utf-8 -*-
# @Time    : 12/9/2022
# @Author  : Jing Zhang

import numpy as np
from numpy.linalg import norm
from tqdm import tqdm
import json
import os

'''
This script is to find synonyms for medical entities based on pretrained embeddings.
'''


def calculate_word_embedding_similarity(word_emd_dict: dict, threshold, max_num):
    """
    calculate word pair's similarity using cosine
    :param word_emd_dict: embedding dictionary as {(entity, type): embeddings}
    :param threshold: threshold to be a synonym
    :param max_num: maximum number of synonyms for a word to be chosen
    :return: dictionary as {word, [synonym1, synonym2, ...]}
    """
    entities = list(word_emd_dict.keys())

    print('Embedding similarity calculation is started.')
    word_emds = np.array(list(word_emd_dict.values()))
    word_norm = norm(word_emds, axis=1)
    norm_emds = word_emds / word_norm[:, np.newaxis]
    simi_emds = np.dot(norm_emds, norm_emds.transpose())
    simi_emds[simi_emds < threshold] = 0

    simi_indices = simi_emds.argsort(axis=-1)[:, ::-1][:, 1:max_num + 1]
    simi_mask = np.take_along_axis(simi_emds, simi_indices, axis=-1)
    simi_mask[simi_mask > 0] = True
    simi_mask[simi_mask <= 0] = False

    real_num = simi_mask.sum(axis=-1)
    print('Embedding similarity calculation is finished.')

    # convert into dictionary for easy access
    synonym_dict = {}
    for i in tqdm(range(len(entities))):
        ent, type_ = entities[i]
        if real_num[i] > 0:
            indices = simi_indices[i][0:int(real_num[i])]
            similar_words_info = [entities[idx] for idx in indices]
            similar_words = [word[0] for word in similar_words_info if word[1] == type_]
            if len(similar_words) > 0:
                synonym_dict[ent] = similar_words

    print(f'{len(synonym_dict)} entities found synonyms based on similarity {threshold}.')
    return synonym_dict


def save_synonym_dict(filename, synonym_dict, sep=' '):
    """
     Write synonyms for medical entities into file in the format "word synonym1 synonym2, ..."
    :param filename: the file path
    :param synonym_dict: dictionary in the format {word, [synonym1, synonym2, ...]}
    :param sep: separator of words in the file
    :return:
    """
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    with open(filename, 'w', encoding='utf-8') as fw:
        for ent, synonym_ent_list in synonym_dict.items():
            synonyms = sep.join(synonym_ent_list)
            fw.write(ent + sep + synonyms)
            fw.write('\n')
        print(f'synonym dictionary is written in {filename}.')


def read_synonym_dict(filename, sep=' '):
    """
    Read synonyms from file in the format "word synonym1 synonym2, ..."
    :param filename: the file path
    :param sep: separator of words in the file
    :return: dictionary in the format {word, [synonym1, synonym2, ...]}
    """
    synonym_dict = {}
    with open(filename, 'r', encoding='utf-8') as fr:
        data = fr.readlines()
        for line in data:
            entities = line.strip()
            if len(entities) == 0:
                continue
            synonym_ent_list = entities.split(sep)
            synonym_dict[synonym_ent_list[0]] = synonym_ent_list[1:]
        print(f'synonym dictionary is read from {filename}.')
    return synonym_dict


def load_embedding_file(emd_file, entity_dict=None, emd_dim=512, sep=' '):
    """
    load pretrained embeddings from file;
    retain only those entities in the entity_dict;
    :param emd_file: embedding file path
    :param entity_dict:  dictionary as {entity: type}
    :param emd_dim:  embedding dimension, which is specified in the embedding file
    :param sep: separator in the embedding file
    :return: entity dictionary as {(entity, type): embeddings}
    """
    with open(emd_file, 'r', encoding='utf-8') as fr:
        data = fr.readlines()
        word_emds = {}

        for line in data:
            line = line.strip().split(sep)
            if len(line) < emd_dim + 1:
                continue

            ent = line[0]
            if entity_dict is not None and ent not in entity_dict:
                continue

            entity_type = entity_dict[ent] if entity_dict is not None else 'None'

            emd = [float(item) for item in line[1:]]
            word_emds[(ent, entity_type)] = emd

        print(f'{len(word_emds)} medical embeddings are found in {emd_file}.')
        return word_emds


def load_entity_from_file(banned_types, *data_files):
    """
    load medical entities from multiple json files;
    this function needs to be revised according to your file format
    :param banned_types: a list of entity types that need to be filtered
    :param data_files: a list of files (in json format)
    :return: a list of entities in tuple (entity, entity_type)
    """
    entity_list = []
    for filename in data_files:
        with open(filename, 'r', encoding='utf-8') as fr:
            samples = json.load(fr)
            for sample in samples:
                entities = sample['entities']
                for entity in entities:
                    if entity['type'] not in banned_types:
                        entity_list.append((entity['entity'], entity['type']))

    entity_set = set(entity_list)
    print(f'{len(entity_set)} entities are loaded.')
    return list(entity_set)


if __name__ == '__main__':
    entities = load_entity_from_file(['sym'], 'Corpus/CMeEE_train.json', 'Corpus/CMeEE_dev.json')
    emd_dict = load_embedding_file(emd_file="WordEmbeddings/medical_embeddings.txt",
                                   entity_dict=dict(entities),
                                   emd_dim=512,
                                   sep=' ')
    threshold = 0.7
    synonym_dict = calculate_word_embedding_similarity(word_emd_dict=emd_dict,
                                                       threshold=threshold,
                                                       max_num=4)
    save_synonym_dict(filename=f'Synonym/medical_synonyms_{threshold}.txt',
                      synonym_dict=synonym_dict,
                      sep=' ')
    # show results
    for word, words in synonym_dict.items():
        print(f'{word}: ' + ' '.join(words))
