import math
from functools import reduce
from itertools import groupby, starmap

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sn


global avg_map
global dev_exp_map
global avg_map_by_gene
global top_500_genes
global top_500_genes_map
global centroids
global val_map
global idx_avg


# Normalizacija unutar celija


def average_gene_exp1(array, x):
    # print(array)
    if array and array[-1][0] == x[0]:
        array[-1] = array[-1][0], (array[-1][1] + x[2])/2
    else:
        array.append((x[0], x[2]))
    return array


def average_gene_exp(array, x):
    if array and array[0][1][0] == x[0]:
        # print(array[0][1][0])
        array[0] = list(array[0])
        array[0][1] = [array[0][1][0], array[0][1][1] + x[2], array[0][1][2] + 1]
        array[0] = tuple(array[0])
    else:
        if array:
            # print(array)
            array[0] = list(array[0])
            array[0][0].append((array[0][1][0], array[0][1][1]/array[0][1][2]))
            array[0][1] = [x[0], x[2], 1]
            array[0] = tuple(array[0])
        else:
            array.append(([], [x[0], x[2], 1]))
    return array


def center_exp_val(tuple_in):
    cell, gene, val = tuple_in
    return cell, gene, val-avg_map[cell]


def variance_of_exp(array, x):
    # print(x)
    # if len(array) == 1 or len(array) == 0:
    # print(array)
    if array and array[0][1][0] == x[0]:
        # print(array[0][1][0])
        array[0] = list(array[0])
        array[0][1] = [array[0][1][0], array[0][1][1] + x[2] ** 2, array[0][1][2] + 1]
        array[0] = tuple(array[0])
    else:
        if array:
            # print(array)
            array[0] = list(array[0])
            array[0][0].append((array[0][1][0], array[0][1][1]/array[0][1][2]))
            array[0][1] = [x[0], x[2], 1]
            array[0] = tuple(array[0])
        else:
            array.append(([], [x[0], x[2], 1]))
    return array


def post_proc(array):
    last = array[0][1]
    last = last[0], last[1] / last[2]
    array[0] = list(array[0])
    array[0][0].append(last)
    return array[0][0]


def deviation_of_exp(tuple_in):
    cell, val = tuple_in
    return cell, math.sqrt(val)


def standard_exp(tuple_in):
    cell, gene, val = tuple_in
    return cell, gene, val/dev_exp_map[cell]


def avg_gene_exp_by_gene(array, x):
    if array and array[-1][0] == x[1]:
        array[-1] = array[-1][0], (array[-1][1] + x[2])/2
    else:
        array.append((x[1], x[2]))
    return array


def center_exp_val_by_gene(tuple_in):
    cell, gene, val = tuple_in
    return cell, gene, val-avg_map_by_gene[gene]


def variance_exp_by_gene(array, x):
    # print(x)
    if array and array[0][1][0] == x[1]:
        array[0] = list(array[0])
        array[0][1] = [array[0][1][0], array[0][1][1] + x[2] ** 2, array[0][1][2] + 1]
        array[0] = tuple(array[0])
    else:
        if array:
            # print(array) st(array[0])
            array[0] = list(array[0])
            array[0][0].append((array[0][1][0], array[0][1][1]/array[0][1][2]))
            array[0][1] = [x[1], x[2], 0]
            array[0] = tuple(array[0])
        else:
            array.append(([], [x[1], x[2], 0]))
    return array


def top_500_var_by_gene(array, x):
    if array and array[0][1] == 500:
        return array
    if array:
        array[0] = list(array[0])
        array[0][0].append(x[0])
        array[0][1] = array[0][1] + 1
        array[0] = tuple(array[0])
    else:
        array.append(([], 0))
        array[0] = list(array[0])
        array[0][0].append(x[0])
        array[0][1] = array[0][1] + 1  # moglo je lakse ali ovako je razumljivije
        array[0] = tuple(array[0])

    return array


def post_proc_by_gene(array):
    return array[0][0]


def filter_std_by_top_500_gene(array, x):
    if x[1] in top_500_genes:
        array.append((x[0], x[1], x[2]))
    return array


def normalize_rank(array, x):
    if array and array[-1][1] == x[1]:
        if array[-1][2] == x[2]:
            array.append((x[0], x[1], x[2], array[-1][3]))
        else:
            array.append((x[0], x[1], x[2], array[-1][3] + 1))
    else:
        array.append((x[0], x[1], x[2], 1))

    return array


# def highest_var_without_original_val(array):
#     return array[]

def top_500_var_by_gene_modified(array, x):
    if array and array[0][1] == 500:
        return array
    if array:
        array[0] = list(array[0])
        array[0][0].append((x[0], x[1]))
        array[0][1] = array[0][1] + 1
        array[0] = tuple(array[0])
    else:
        array.append(([], 0))
        array[0] = list(array[0])
        array[0][0].append((x[0], x[1]))
        array[0][1] = array[0][1] + 1  # moglo je lakse ali ovako je razumljivije
        array[0] = tuple(array[0])

    return array


def filter_std_by_top_500_gene_modified(array, x):
    if x[1] in top_500_genes:
        array.append((x[0], x[1], x[2]))
    return array


def remove_original_val_from_top_500(array):
    return array[0]


def groups_to_dict(array):
    return {k: [x[2] for x in v] for k, v in array}


def list_transfer(x):
    return list(x)


def coord_extract(x):
    return x[1], x[2]


def calc_which_cluster_point_belongs_to(array, x):
    distance = np.sqrt(((centroids - x[1]) ** 2).sum(axis=1))
    array.append((x[0], np.argmin(distance)))
    return array


def reduce_group_by_cells(array, x):
    if array and array[-1][0] == x[1]:
        array[-1][1].append(x[0])
    else:
        array.append((x[1], [x[0]]))

    return array


def kv_reduce(x, y):
    # print(reduce(add, y, 0))
    # print(y)
    # print(len(y))
    byproduct = reduce(add, y, 0)
    return x, (byproduct[0]/len(y), byproduct[1]/len(y))


def add(x, y):
    if x:
        x = list(x)
        x[0] = x[0] + val_map[y][0]
        x[1] = x[1] + val_map[y][1]
        x = tuple(x)
    else:
        x = val_map[y]
    return x


def update_centroids(array, x):
    # print("HURAYYY")
    # print(array)
    # print(x)
    if array:
        array[0] = list(array[0])
        if array[0][1] in idx_avg.keys():
            array[0][0].append(np.array(idx_avg[array[0][1]]))
        else:
            array[0][0].append(x)
        array[0][1] = array[0][1]+1
        array[0] = tuple(array[0])
    else:
        if 0 in idx_avg.keys():
            # print("ALAAAAAA")
            # print(idx_avg[0])
            array.append(([np.array(idx_avg[0])], 1))
        else:
            array.append(([x], 1))

    return array


def post_proc_final(array):
    return array[0][0]


def main():
    df = pd.read_table('ekspresije.tsv', index_col=0)
    data = [(cell, gene, value) for cell in df.columns
            for gene, value in df[cell].items()]
    # print(data)
    average = reduce(average_gene_exp, data, [])
    # print(average)
    average = post_proc(average)
    # print(average)

    global avg_map
    avg_map = dict(average)
    # print(avg_map)

    centered_exp_val = list(map(center_exp_val, data))
    # print(centered_exp_val)

    var_exp = reduce(variance_of_exp, centered_exp_val, [])
    var_exp = post_proc(var_exp)
    # print(var_exp)

    dev_exp = list(map(deviation_of_exp, var_exp))
    # print(dev_exp)

    global dev_exp_map
    dev_exp_map = dict(dev_exp)

    std_exp = list(map(standard_exp, centered_exp_val))
    # print(std_exp)

    # 2. Nomralizacija vrednosti ekspresije gena
    gene_name_sorted = sorted(std_exp, key=lambda x: x[1])
    # print(gene_name_sorted)

    avg_gene = reduce(avg_gene_exp_by_gene, gene_name_sorted, [])
    global avg_map_by_gene
    avg_map_by_gene = dict(avg_gene)

    centered_exp_val_by_gene = list(map(center_exp_val, gene_name_sorted))
    # print(centered_exp_val_by_gene)

    var_exp_by_gene = reduce(variance_exp_by_gene, centered_exp_val_by_gene, [])
    # print(var_exp_by_gene)
    var_exp_by_gene = post_proc(var_exp_by_gene)
    # print(var_exp_by_gene)

    highest_var_by_gene = reduce(top_500_var_by_gene, sorted(var_exp_by_gene, key=lambda x: x[1], reverse=True), [])
    # print(highest_var_by_gene)
    highest_var_by_gene = post_proc_by_gene(highest_var_by_gene)
    # print(highest_var_by_gene)
    # print(len(highest_var_by_gene))

    global top_500_genes
    top_500_genes = highest_var_by_gene

    # print(len(std_exp))
    top_500_genes_and_cells = reduce(filter_std_by_top_500_gene, std_exp, [])
    # print(top_500_genes_and_cells)
    # print(len(top_500_genes_and_cells))
    sorted_by_gene_and_val = sorted(top_500_genes_and_cells, key=lambda x: (x[1], x[2]))
    # print(sorted_by_gene_and_val)

    rank_normalized_gene = reduce(normalize_rank, sorted_by_gene_and_val, [])
    # print(len(rank_normalized_gene))
    # print(rank_normalized_gene)

    # pogresno shvacen 2.2 i zbog toga sve od 2.2 mora da se menja ali gore ostavljena prosla implementacija

    highest_var_by_gene = reduce(top_500_var_by_gene_modified, sorted(var_exp_by_gene, key=lambda x: x[1], reverse=True), [])
    highest_var_by_gene = post_proc_by_gene(highest_var_by_gene)

    global top_500_genes_map
    top_500_genes_map = dict(highest_var_by_gene)
    # print(top_500_genes_map)
    # print(std_exp)

    top_500_genes_and_cells = reduce(filter_std_by_top_500_gene_modified, std_exp, [])
    # print(top_500_genes_and_cells)
    # print(len(std_exp))
    # print(len(top_500_genes_and_cells))

    sorted_by_gene_and_val = sorted(top_500_genes_and_cells, key=lambda x: (x[1], x[2]))
    # print(sorted_by_gene_and_val)
    rank_normalized_gene = reduce(normalize_rank, sorted_by_gene_and_val, [])
    # print(rank_normalized_gene)
    # print(highest_var_by_gene)

    top_500_higest_var_genes_without_original_val = list(map(remove_original_val_from_top_500, highest_var_by_gene))
    # print(top_500_higest_var_genes_without_original_val)

    # print(sorted_by_gene_and_val)

    # 3. K-means klasterovanje

    grouped_normalized_cells = groupby(sorted(rank_normalized_gene, key=lambda x: (x[0], x[1])), key=lambda x: x[0])
    grouped_normalized_cells = groups_to_dict(grouped_normalized_cells)
    # print(grouped_normalized_cells)

    embedding = pd.read_table('umap.tsv')
    embedding['cluster'] = 0  # embedding['cluster'] = embedding.cell.map(klasteri) za klasteri = {'cell_id': 'cluster_id'}

    k = [2, 4, 6, 10]
    np.random.seed(15)
    for val in k:
        global centroids
        centroids = np.random.uniform(min(min(embedding['umap1']), min(embedding['umap2'])), max(max(embedding['umap1']), max(embedding['umap2'])), (val, 2))
        # print(centroids)
        cell_name = list(embedding['cell'])
        coord_x = list(embedding['umap1'])
        coord_y = list(embedding['umap2'])
        list_of_vals = list(zip(cell_name, zip(coord_x, coord_y)))
        # print(list_of_vals)
        global val_map
        val_map = dict(list_of_vals)
        for iteration in range(250):
            new_map = reduce(calc_which_cluster_point_belongs_to, list_of_vals, [])
            # print(new_map)

            # print(grouped)

            if iteration == 0 or iteration == 9 or iteration == 249:
                embedding['cluster'] = embedding.cell.map(dict(new_map))
                # print(embedding['cluster'])
                plt.figure(figsize=(8, 6))
                for i, c in enumerate(centroids):
                    plt.scatter(centroids[i][0], centroids[i][1], marker='*', s=150, c=sn.color_palette()[i])
                plt.scatter(
                    embedding.umap1,
                    embedding.umap2,
                    c=[sn.color_palette()[x] for x in embedding.cluster]
                )
                plt.show()
                grouped = reduce(reduce_group_by_cells, sorted(new_map, key=lambda x: x[1]), [])
                # print(grouped)
                grouped = dict(grouped)
                # print(grouped.items())
                global idx_avg
                idx_avg = list(starmap(kv_reduce, grouped.items()))
                # print(idx_avg)
                idx_avg = dict(idx_avg)
                centroids = list(reduce(update_centroids, centroids, []))
                # print(centroids)
                centroids = np.array(post_proc_final(centroids))
                # print(centroids)

    # print(embedding)


if __name__ == '__main__':
    main()

