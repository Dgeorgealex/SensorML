from rdflib import Graph, Namespace
from owlrl import DeductiveClosure, OWLRL
from rdflib.plugins.sparql import prepareQuery
import os


def make_inference(parameters):
    script_directory = os.path.dirname(os.path.abspath(__file__))
    relative_file_path = '../assets/tomatoes.rdf'
    absolute_file_path = os.path.join(script_directory, relative_file_path)
    queryDict = {
        "part": "hasAtPart",
        "intensity": "hasIntensity",
        "texture": "hasTexture",
        "color": "hasColor",
        "pattern": "hasPattern",
        "anatomicalRegion": "hasAtPart",
        "shape": "hasShape",
        "borderColor": "hasBorderColor"
    }

    if parameters.get("anatomicalRegion") and parameters.get("part"):
        del parameters["part"]

    abnormalities = ""

    for key, value in parameters.items():
        if key in queryDict.keys():
            abnormalities += f"ex:{queryDict[key]} ex:{value} ;\n"

    abnormalities = abnormalities.rsplit(';', 1)[0] + '.'

    query_str = f"""
        PREFIX ex: <http://www.semanticweb.org/cezar/ontologies/2024/0/tomatoes#>

        SELECT ?disease
        WHERE {{
            ?abnormalityGroup rdf:type ex:AbnormalityGroup ;
                                {abnormalities}
            ?disease ex:hasAbnormalityGroup ?abnormalityGroup .
        }}
    """

    g = Graph()
    g.parse(absolute_file_path, format="xml")  # Adjust the format as needed
    DeductiveClosure(OWLRL.OWLRL_Semantics).expand(g)

    query = prepareQuery(query_str, initNs=dict(g.namespaces()))

    results = []
    for row in g.query(query):
        results.append(row['disease'].split('#')[1])
    return set(results)
