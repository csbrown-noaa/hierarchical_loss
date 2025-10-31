import requests


WORMS_TREE_URL = 'https://www.marinespecies.org/rest/AphiaClassificationByAphiaID/{}'
WORMS_NAME_URL = 'https://www.marinespecies.org/rest/AphiaNameByAphiaID/{}'
WORMS_ID_URL = 'https://www.marinespecies.org/rest/AphiaIDByName/{}?marine_only=true&extant_only=true'


def get_WORMS_id(name: str) -> int:
    """Fetches the AphiaID from WORMS for a given scientific name.

    Parameters
    ----------
    name : str
        The scientific name of the organism to look up.

    Returns
    -------
    int
        The corresponding AphiaID from the WORMS database.

    Examples
    --------
    >>> get_WORMS_id('Gnathostomata')
    1828
    """
    result = requests.get(WORMS_ID_URL.format(name))
    return int(result.content)


def get_WORMS_name(WORMS_id: int) -> str:
    """Fetches the scientific name from WORMS for a given AphiaID.

    The returned name is stripped of the surrounding double quotes
    that the API returns.

    Parameters
    ----------
    WORMS_id : int
        The AphiaID of the organism to look up.

    Returns
    -------
    str
        The corresponding scientific name.

    Examples
    --------
    >>> get_WORMS_name(1828)
    'Gnathostomata'
    """
    result = requests.get(WORMS_NAME_URL.format(WORMS_id))
    return result.content.decode("utf-8")[1:-1]

def get_WORMS_tree(organism_id: int | str) -> dict:
    """Fetches the full hierarchical classification tree from WORMS.

    Given an AphiaID or scientific name, retrieves the classification
    hierarchy from the root ("Biota") down to the specified organism.

    Parameters
    ----------
    organism_id : int | str
        The AphiaID or scientific name of the organism.

    Returns
    -------
    dict
        A nested dictionary representing the classification tree.

    Examples
    --------
    >>> import json
    >>> print(json.dumps(get_WORMS_tree(get_WORMS_id('Gnathostomata')), indent=4))
    {
        "AphiaID": 1,
        "rank": "Superdomain",
        "scientificname": "Biota",
        "child": {
            "AphiaID": 2,
            "rank": "Kingdom",
            "scientificname": "Animalia",
            "child": {
                "AphiaID": 1821,
                "rank": "Phylum",
                "scientificname": "Chordata",
                "child": {
                    "AphiaID": 146419,
                    "rank": "Subphylum",
                    "scientificname": "Vertebrata",
                    "child": {
                        "AphiaID": 1828,
                        "rank": "Infraphylum",
                        "scientificname": "Gnathostomata",
                        "child": null
                    }
                }
            }
        }
    }
    """
    result = requests.get(WORMS_TREE_URL.format(organism_id))
    return result.json()


def WORMS_tree_to_childparent_tree(worms_trees: list[dict]) -> dict[int, int]:
    """Converts one or more WORMS classification trees into a {child: parent} dict.

    This function processes a list of nested tree structures (as returned by
    `get_WORMS_tree`) and flattens them into a single dictionary that maps
    each child AphiaID to its immediate parent AphiaID.

    Parameters
    ----------
    worms_trees : list[dict]
        A list of nested tree structures from the WORMS API.

    Returns
    -------
    dict[int, int]
        A single dictionary representing the hierarchy in
        {child_AphiaID: parent_AphiaID} format.

    Examples
    --------
    >>> tree1 = {
    ...   "AphiaID": 1, "scientificname": "Biota", "child": {
    ...     "AphiaID": 2, "scientificname": "Animalia", "child": {
    ...       "AphiaID": 1821, "scientificname": "Chordata", "child": None
    ...     }
    ...   }
    ... }
    >>> tree2 = {
    ...     "AphiaID": 1,
    ...     "rank": "Superdomain",
    ...     "scientificname": "Biota",
    ...     "child": {
    ...         "AphiaID": 2,
    ...         "rank": "Kingdom",
    ...         "scientificname": "Animalia",
    ...         "child": {
    ...             "AphiaID": 1065,
    ...             "rank": "Phylum",
    ...             "scientificname": "Arthropoda",
    ...             "child": {
    ...                 "AphiaID": 1274,
    ...                 "rank": "Subphylum",
    ...                 "scientificname": "Chelicerata",
    ...                 "child": {
    ...                     "AphiaID": 1300,
    ...                     "rank": "Class",
    ...                     "scientificname": "Arachnida",
    ...                     "child": None
    ...                 }
    ...             }
    ...         }
    ...     }
    ... }
    >>> WORMS_tree_to_childparent_tree([tree1, tree2])
    {2: 1, 1821: 2, 1065: 2, 1274: 1065, 1300: 1274}
    """
    childparent_tree = {}
    for tree in worms_trees:
        try:
            parent = tree['AphiaID']
        except Exception as e:
            print("could not find id")
            print(tree)
            raise e
        while 'child' in tree and tree['child']:
            tree = tree['child']
            try:
                child = tree['AphiaID']
            except Exception as e:
                print("could not find id")
                print(tree)
                raise e
            childparent_tree[child] = parent
            parent = child
    return childparent_tree

        
