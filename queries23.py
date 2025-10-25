from owlready2 import get_ontology, default_world
from typing_extensions import TypedDict
from pydantic import BaseModel
import re
import ollama

# ============================================================
# Data Structures
# ============================================================

class InputState(TypedDict):
    story: str

class InbetweenState(TypedDict):
    story: str
    chunks: list[str]
    ontology_summary: str
    extract: dict
    conflicts: dict
    rewritten_chunks: list[str]
    re_verified_chunks: list[str]
    evaluated_chunks: list[str]

class OutputState(TypedDict):
    revised_story: str

class Triple(BaseModel):
    subj: str
    pred: str
    obj: str

# ============================================================
# Load Ontology
# ============================================================

ONTOLOGY_PATH = "/Users/ahlam/Downloads/IA_ontology4.owl"
onto = get_ontology(ONTOLOGY_PATH).load()

print("=== Ontology Loaded ===")
print("Classes:", [c.name for c in onto.classes()])
print("Individuals:", [i.name for i in onto.individuals()])
print("Object Properties:", [p.name for p in onto.object_properties()])
print("Data Properties:", [p.name for p in onto.data_properties()])


# ============================================================
# Step 1: Chunk Story
# ============================================================

def chunk(state: InputState) -> InbetweenState:
    sentences = re.split(r'(?<=[.!?])\s+', state['story'].strip())
    chunks = [s.strip() for s in sentences if s.strip()]
    
    print("\n=== Node: Chunk ===")
    print(chunks)

    return {
        "story": state['story'],
        "chunks": chunks,
        "ontology_summary": "",
        "extract": {},
        "conflicts": {},
        "rewritten_chunks": [],
        "re_verified_chunks": [],
        "evaluated_chunks": []
    }
# ============================================================
# Step 2: Extract Triples using regex
# ============================================================

def extract(state: InbetweenState) -> InbetweenState:
    """
    Extract triples from story chunks, handling pronouns and flexible phrasing.
    Ensures marriage, age, allergies, and consumed food are captured.
    """
    print("\n=== Node: Extract Triples ===")
    extracts = {}
    previous_subject = None  # Tracks last mentioned person

    def norm_subj(s):
        """Normalizes subject names (e.g., converts 'She'/'He' to previous_subject)."""
        if s in ["She", "He"] and previous_subject:
            return previous_subject
        s = s.title()
        
        individuals = [i.name for i in onto.individuals()]
        
        # Map plural to singular if ontology has that individual
        if s not in individuals and s.endswith('s'):
            s_singular = s[:-1]
            if s_singular in individuals:
                s = s_singular
        
        # keep original if not in ontology
        return s
    
    for i, chunk_text in enumerate(state['chunks']):
        triples = []
        seen_triples = set()  

        # Detect first proper noun as subject (fallback for pronouns)
        match_subj = re.search(r'\b([A-Z][a-z]+)\b', chunk_text)
        if match_subj:
            previous_subject = match_subj.group(1)

        # --- Age extraction ---
        for m in re.finditer(r'\b(She|He|[A-Z][a-z]+)\s+is\s+(\d{1,3})\s+years?\s+old', chunk_text):
            subj = m.group(1)
            if subj in ["She", "He"] and previous_subject:
                subj = previous_subject
            triples.append({"subj": subj, "pred": "hasAge", "obj": m.group(2)})

        # --- Marriage extraction ---
        for m in re.finditer(
            r'\b(She|He|[A-Z][a-z]+)\b.*?married.*?\b([A-Z][a-z]+)\b',
            chunk_text
        ):
            subj = m.group(1)
            if subj in ["She", "He"] and previous_subject:
                subj = previous_subject
            triples.append({"subj": subj, "pred": "isMarriedTo", "obj": m.group(2)})

        #--- City Properties Extraction ---
        for m in re.finditer(r'\b(?:in|at|from)\s+([A-Z][a-z]+)\b.*?\b(quiet|empty|calm|busy|noisy|crowded)\b', chunk_text, flags=re.IGNORECASE):
            city = m.group(1)
            desc = m.group(2).lower()
            t = (city, "description", desc)
            if t not in seen_triples:
                triples.append({"subj": city, "pred": "description", "obj": desc})
                seen_triples.add(t)

        # --- Occupation Extraction ---
        for m in re.finditer(r'\b([A-Z][a-z]+)\b.*?\b(?:works\s+as|works\s+as\s+an|works\s+as\s+a|is\s+a|is\s+an|became)\s+([A-Za-z][A-Za-z\s\-]+?)(?:[.,;]|$)', chunk_text, flags=re.IGNORECASE):
            subj = norm_subj(m.group(1))
            occ = m.group(2).strip()
            # filter out false positives
            if re.search(r'\b(married|allergic|allergy|peanut|peanuts|pizza|sandwich|reserved person)\b', occ, flags=re.IGNORECASE):
                continue
            t = (subj, "hasOccupation", occ)
            if t not in seen_triples:
                triples.append({"subj": subj, "pred": "hasOccupation", "obj": occ})
                seen_triples.add(t)
        
        # --- Character Trait Extraction ---
        for m in re.finditer(
            r'([A-Z][a-z]+)\'s\s+(?:husband|wife|spouse)\s+describes\s+(?:her|him|them)\s+as\s+a?\s+([A-Za-z\s\-]+?)\s+person',
            chunk_text
        ):
            subj = norm_subj(m.group(1))
            trait_raw = m.group(2).strip()
            trait = trait_raw.replace('-', ' ').title().replace(' ', '_')
            
            t = (subj, "hasTrait", trait)
            if t not in seen_triples:
                triples.append({"subj": subj, "pred": "hasTrait", "obj": trait})
                seen_triples.add(t)

        # City with population
        for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+has\s+(?:a\s+)?(high|low|large|small)\s+population\b', chunk_text, flags=re.IGNORECASE):
            subj = m.group(1)
            pop = m.group(2).lower()
            t = (subj, "hasPopulation", pop)
            if t not in seen_triples:
                triples.append({"subj": subj, "pred": "hasPopulation", "obj": pop})
                seen_triples.add(t)


            # Trait extractor
        for m in re.finditer(
                r'\b([A-Z][a-z]+|She|He)\s+is\s+(reserved|social|goodhearted|lazy)',
                chunk_text, flags=re.IGNORECASE
            ):
                subj = norm_subj(m.group(1))
                trait_raw = m.group(2).strip()
                trait = trait_raw.title().replace(' ', '_')

                t = (subj, "hasTrait", trait)
                if t not in seen_triples:
                    triples.append({"subj": subj, "pred": "hasTrait", "obj": trait})
                    seen_triples.add(t)


        for m in re.finditer(
            r'\b' + r'((?:[A-Z][a-zA-Z0-9_]+(?:\s[A-Z][a-zA-Z0-9_]+)*))' + r'\s+is\s+(?:located\s+in|in|at)\s+' + r'((?:[A-Z][a-zA-Z0-9_]+(?:\s[A-Z][a-zA-Z0-9_]+)*))'+ r'\.?',
            chunk_text, flags=re.IGNORECASE
        ):    
            subj_raw = m.group(1).strip()
            obj_raw = m.group(2).strip()
            
            subj_words = subj_raw.split()
            if subj_words and subj_words[0] in ["The", "A", "An"]:
                subj_raw = " ".join(subj_words[1:])
            if not subj_raw:
                continue

            subj = subj_raw.replace(' ', '_')
            obj = norm_subj(obj_raw)   

            t = (subj, "isLocatedIn", obj)
            if t not in seen_triples:
                triples.append({"subj": subj, "pred": "isLocatedIn", "obj": obj})
                seen_triples.add(t)

        # electricity status
        for m in re.finditer(
            r'\b(?:no|without|lack\s+of)\s+([A-Za-z]+)\s+in\s+([A-Z][a-z]+)\b', 
            chunk_text, flags=re.IGNORECASE
        ):
            resource = m.group(1).title() 
            location = m.group(2).title()
            status = 'NoElectricity' if resource == 'Electricity' else f'No{resource}'
            t = (location, "hasResourceStatus", status)
            if t not in seen_triples:
                triples.append({"subj": location, "pred": "hasResourceStatus", "obj": status})
                seen_triples.add(t)
        
        #tool extraction
        for m in re.finditer(
            r'\b(croissants|baguettes|cake|macaron|food)\s+.*?\b(?:made|prepared)\s+in\s+the\s+(oven|stove|pan)\b', 
            chunk_text, flags=re.IGNORECASE
        ):
            food_item = m.group(1)
            tool = m.group(2).title() 

            #map the tool to the activity
            if tool == 'Oven':
                activity = 'Baking'
            elif tool == 'Stove':
                activity = 'Cooking'
            else:
                activity = 'Activity'
            
            t = (activity, "requiresTool", tool)
            if t not in seen_triples:
                triples.append({"subj": activity, "pred": "requiresTool", "obj": tool})
                seen_triples.add(t)


        extracts[f"chunk_{i}"] = triples
        print(f"chunk_{i} -> {triples}")

    state['extract'] = extracts
    return state


# ============================================================
# Step 3: Check Conflicts Using SPARQL Queries
# ============================================================

def check_conflicts(state: InbetweenState) -> InbetweenState:
    print("\n=== Node: Check Conflicts with SPARQL ===")
    conflicts = {}

    PREFIX_LINE = f"PREFIX ex: <http://IA.org/onto.owl#>"

    for chunk_name, triples in state['extract'].items():
        chunk_conflicts = []


        for t in triples:
            q = ""
            
            if t['pred'] in ["hasAge", "isMarriedTo"]:
                q = f"""
                {PREFIX_LINE}
                SELECT ?person WHERE {{
                    ?person a ex:Person ;
                            ex:hasAge ?age ;
                            ex:isMarriedTo ?spouse .
                    FILTER(xsd:integer(?age) = 17)
                }}
                """
            elif t['pred'] == "isLocatedIn":
                # Query naar de echte locatie uit de ontology
                q = f"""
                PREFIX ex: <http://IA.org/onto.owl#>
                SELECT ?loc WHERE {{
                    BIND(ex:{t['subj']} AS ?ind) .
                    ?ind ex:isLocatedIn ?loc .
                }}
                """


                try:
                    results = list(default_world.sparql(q, error_on_undefined_entities=False))
                    if results:
                        # Haal echte waarde uit de ontology
                        real_locations = [str(r[0]).split("#")[-1] for r in results]  
                        if t['obj'] not in real_locations:
                            print(f"Conflict: {t['subj']} isLocatedIn {t['obj']}, ontology says {real_locations}")
                            # Corrigeer triple naar juiste waarde
                            t['obj'] = real_locations[0]
                            chunk_conflicts.append(t)
                    else:
                        print(f"No ontology data for {t['subj']}")
                        chunk_conflicts.append(t)
                except Exception as e:
                    print(f"Error checking location for {t['subj']}: {e}")
                    chunk_conflicts.append(t)

            elif t['pred'] == "hasTrait":
                q = f"""
                PREFIX ex: <http://IA.org/onto.owl#>
                SELECT ?loc WHERE {{
                    BIND(ex:{t['subj']} AS ?ind) .
                    ?ind ex:hasTrait ?loc .
                }}
                """
                
                try:
                    results = list(default_world.sparql(q, error_on_undefined_entities=False))
                    
                    if results:
                        real_traits = [str(r[0]).split("#")[-1] for r in results] 
                        if t['obj'] not in real_traits:
                            print(f"Conflict: {t['subj']} hasTrait {t['obj']}, ontology says {real_traits}")
                            chunk_conflicts.append(t)
                    else:
                        print(f"No ontology data for {t['subj']}")
                        chunk_conflicts.append(t)
                        
                except Exception as e:
                    print(f"Error checking trait for {t['subj']}: {e}")
                    chunk_conflicts.append(t)


            elif t['pred'] == "description" and t['subj'] == "Paris":
                # Only check for conflicts if the description implies quietness/smallness
                if t['obj'] in ["quiet", "empty", "calm"]:
                   q = f"""
                    {PREFIX_LINE}
                    SELECT ?city WHERE {{
                        BIND(ex:Paris AS ?city) .
                        ?city a ex:BusyCity .
                    }}
                    """
            elif t['pred'] == "hasResourceStatus" and t['obj'] == "NoElectricity":
                q = """
                PREFIX ex: <http://IA.org/onto.owl#>
                SELECT ?loc WHERE {
                    ex:Oven ex:requiresResource ?loc .
                }
                """
                try:
                    results = list(default_world.sparql(q, error_on_undefined_entities=False))
                    if results:
                        # Extract actual ontology values
                        real_locations = [str(r[0]).split("#")[-1] for r in results]
                        if t['obj'] not in real_locations:
                            print(f"Conflict: Ontology says Oven requiresResource {real_locations}, "
                                f"but {t['subj']} {t['pred']} {t['obj']}")
                            # Correct triple to match ontology
                            t['obj'] = real_locations[0]
                            chunk_conflicts.append(t)
                    else:
                        print("No ontology data for Oven")
                        chunk_conflicts.append(t)
                except Exception as e:
                    print(f"Error checking location for Oven: {e}")
                    chunk_conflicts.append(t)

                
            
            elif t['pred'] == "hasOccupation":
                q = f"""
                {PREFIX_LINE}
                SELECT ?person WHERE {{
                    ?person a ex:Person ;
                            ex:hasOccupation ?occupation .
                }}
                GROUP BY ?person
                HAVING (COUNT(DISTINCT ?occupation) > 1)
                """
            
            elif t['pred'] == "hasTrait":
                # If a person already has a trait in the ontology and the story suggests a new one, flag it.
                q = f"""
                {PREFIX_LINE}
                SELECT ?person ?existingTrait WHERE {{
                    BIND(ex:{t['subj']} AS ?person) .
                    ?person ex:hasTrait ?existingTrait .
                    FILTER(STR(?existingTrait) != "{t['obj']}")
                }}
                """
        
                        
            elif t['pred'] == "hasAge" and t['obj'] == "110":
                q = f"""
                {PREFIX_LINE}
                SELECT ?person WHERE {{
                    ?person a ex:Person ;
                            ex:hasAge 110 .
                }}
                """
            
            elif t['pred'] == "description" and t['obj'] == "quiet":
                q = f"""
                {PREFIX_LINE}
                SELECT ?city WHERE {{
                    ?city a ex:City ;
                          ex:hasName "{t['subj']}" ;
                          ex:hasPopulation "high" .
                }}
                """
        
            elif t['pred'] == "hasTrait":
                q = f"""
                {PREFIX_LINE}
                SELECT ?person WHERE {{
                    ?person a ex:Person ;
                            ex:hasTrait ?trait1 ;
                            ex:hasTrait ?trait2 .
                    FILTER(?trait1 != ?trait2)
                    FILTER(STR(?person) = "{t['subj']}")
                }}
                """

                
            elif t['pred'] == "requiresTool":
                q = f"""
                {PREFIX_LINE}
                SELECT ?method WHERE {{
                    ?method a ex:CookingMethod ;
                            ex:requiresTool ?tool .
                    FILTER(STR(?method) = "{t['subj']}" && STR(?tool) = "{t['obj']}")
                }}
                """

            elif t['pred'] == "usesTool":
                q = f"""
                {PREFIX_LINE}
                SELECT ?occupation WHERE {{
                    ?occupation a ex:Occupation ;
                            ex:requiresTool ex:{t['obj'].capitalize()} .
                }}
                """

            # Run query
            try:
                results = list(default_world.sparql(q))
                if not results:
                    chunk_conflicts.append(t)
            except Exception as e:
                print(f"Error checking triple {t}: {e}")
                chunk_conflicts.append(t)

        conflicts[chunk_name] = chunk_conflicts
        print(f"{chunk_name} conflicts -> {chunk_conflicts}")

    state['conflicts'] = conflicts
    return state

# ============================================================
# Step 4: Rewrite Conflicts
# ============================================================

def rewrite_chunk(chunk_text, conflicts_for_chunk):
    if not conflicts_for_chunk:
        return chunk_text
    summary = "\n".join([f"{c['subj']} {c['pred']} {c['obj']}" for c in conflicts_for_chunk])
    prompt = f"""
You are a highly constrained, specialized text editor. Your sole task is to fix logical inconsistencies in a story chunk based on provided triples. You must only return the final rewritten chunk

Story chunk:
{chunk_text}

Detected inconsistencies:
{summary}

First Analyze the detected inconsistencies. 
Then, rewrite the chunk to remove all inconsistencies.
If the inconsistency is in the age, change the age to be higher only if its lower than 110 otherwise decrease.
If the consistency is something exceeding a limit (for example occupations) you MUST resolve it by deleting the excess so it falls within limit
Else you can remove words or add them (LIKE NEGATION) to make the scenario consistent
Remember to only change the inconsistency, and nothing else. DO NOT remove any parts that are not inconsistent
CRUCIAL: You must ONLY return the FINAL rewritten story chunk. Do not include the inconsistency summary, any reasoning, any analysis, or any introductory/explanatory phrases.**
"""




    resp = ollama.chat(model="llama3:latest",
                       messages=[{"role": "user", "content": prompt}])
    return resp.message.content.strip()


def rewrite_conflicts(state: InbetweenState) -> InbetweenState:
    print("\n=== Node: Rewrite Conflicts ===")
    rewritten = []
    for i, chunk in enumerate(state['chunks']):
        name = f"chunk_{i}"
        rewritten_chunk = rewrite_chunk(chunk, state['conflicts'].get(name, []))
        rewritten.append(rewritten_chunk)
        print(f"{name}: {rewritten_chunk}")
    state['rewritten_chunks'] = rewritten
    return state

# ============================================================
# Step 5: Re-verify
# ============================================================

def reverify_chunks(state: InbetweenState) -> InbetweenState:
    print("\n=== Node: Re-verify ===")
    
    temp_state = {
        "story": " ".join(state['rewritten_chunks']),
        "chunks": state['rewritten_chunks'],
        "ontology_summary": "",
        "extract": {},
        "conflicts": {},
        "rewritten_chunks": [],
        "re_verified_chunks": [],
        "evaluated_chunks": []
    }
    
    temp_state = extract(temp_state)
    temp_state = check_conflicts(temp_state)
    
    
    total_conflicts = sum(len(conflicts) for conflicts in temp_state['conflicts'].values())
    
    
    state['re_verified_chunks'] = state['rewritten_chunks']
    return state
   
# ============================================================
# Step 6: Assemble Final Story
# ============================================================

def assemble_final_story(state: InbetweenState) -> OutputState:
    final_story = " ".join(state['re_verified_chunks'])
    print("\n=== Node: Final Story ===")
    print(final_story)
    return {"revised_story": final_story}

# ============================================================
# Run Full Pipeline
# ============================================================

if __name__ == "__main__":
    

    # Scenario 1
    story_text = """
    Amelia is one of France's most famous lawyers. She is 17 years old and recently married her husband Adam, who is 25.
    """


    

    '''
    # Scenario 2
    story_text = """
    They live together in Paris, a quiet city without much hustle and bustle since the city has a large population. 
    The city is located on flat land, so there are no mountains.
    """
    '''
    
    

    '''
    # Scenario 3
    story_text = """ 
    The Eiffel Tower is located in Italy.
   """
   '''

   
  
    '''
    # Scenario 4 
    story_text= """
    Her husband Adam is a baker and a merchant. He makes various types of French pastries, including macarons, but also baguettes and croissants. 
    Unfortunately, Amelia cannot eat many sweets because she is allergic to the filling which contains dairy.
    """
    '''
    
    
    '''
    #scenario 5
    story_text = """
    Amelia's husband describes her as a self-serving and good hearted person who cares about the well-being of people.
    """
    '''
    
    
    '''
    #scenario 6
    story_text = """
    It was the day that there was no electricity in Paris. Amelia went to the bakery were Adam worked.
    When Amelia came in, she immediately bought two croissants which were made in the oven.
    """
    '''
    
    
    '''
    # Scenario 8
    story_text = """
    Los Angeles has a high population and is a quiet and empty city 
    """
    '''
    

    '''
    # Scenario 9
    story_text = """
    Jan is 115 years old. 
    """
    '''
    
    
    '''
    # Scenario 11
    story_text = """
    Jan is reserved. During lunch Jan enjoys conversations with other patients, nurses, and all the other staff. 
   """
   '''
 
  
    state = chunk({"story": story_text})
    state = extract(state)
    state = check_conflicts(state)
    state = rewrite_conflicts(state)
    state = reverify_chunks(state)
    final_state = assemble_final_story(state)

    print("\n=== FINAL OUTPUT ===")
    print(final_state["revised_story"])