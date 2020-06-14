tacred_relations = {
    0: 'org:alternate_names',
    1: 'org:city_of_headquarters',
    2: 'org:country_of_headquarters',
    3: 'org:dissolved',
    4: 'org:founded',
    5: 'org:founded_by',
    6: 'org:member_of',
    7: 'org:members',
    8: 'org:number_of_employees/members',
    9: 'org:parents',
    10: 'org:political/religious_affiliation',
    11: 'org:shareholders',
    12: 'org:stateorprovince_of_headquarters',
    13: 'org:subsidiaries',
    14: 'org:top_members/employees',
    15: 'org:website',
    16: 'per:age',
    17: 'per:alternate_names',
    18: 'per:cause_of_death',
    19: 'per:charges',
    20: 'per:children',
    21: 'per:cities_of_residence',
    22: 'per:city_of_birth',
    23: 'per:city_of_death',
    24: 'per:countries_of_residence',
    25: 'per:country_of_birth',
    26: 'per:country_of_death',
    27: 'per:date_of_birth',
    28: 'per:date_of_death',
    29: 'per:employee_of',
    30: 'per:origin',
    31: 'per:other_family',
    32: 'per:parents',
    33: 'per:religion',
    34: 'per:schools_attended',
    35: 'per:siblings',
    36: 'per:spouse',
    37: 'per:stateorprovince_of_birth',
    38: 'per:stateorprovince_of_death',
    39: 'per:stateorprovinces_of_residence',
    40: 'per:title',
    41: 'no_relation',
}

tacred_rules = [
    (35, 35, 35), # (per:siblings, per:siblings, per:siblings)
    (17, 17, 17), # (per:alternate_names, per:alternate_names, per:alternate_names)

    (17, 35, 35), # (per:alternate_names, per:siblings, per:siblings)
    (35, 17, 35), # (per:per:siblings, per:alternate_names, per:siblings)
    (35, 35, 17), # (per:siblings, per:siblings, per:alternate_names)
    
    (17, 20, 20), # (per:alternate_names, per:children, per:children)
    (20, 17, 20), # (per:children, per:alternate_names, per:children)
    (20, 20, 17), # (per:children, per:children, per:alternate_names)

    (17, 32, 32), # (per:alternate_names, per:parents, per:parents)
    (32, 17, 32), # (per:parents, per:alternate_names, per:parents)
    (32, 32, 17), # (per:parents, per:parents, per:alternate_names)

    (17, 20, 32), # (per:alternate_names, per:children, per:parents)
    (20, 17, 32), # (per:children, per:alternate_names, per:parents)
    (20, 32, 17), # (per:children, per:parents, per:alternate_names)

    (17, 32, 20), # (per:alternate_names, per:parents, per:children)
    (32, 17, 20), # (per:parents, per:alternate_names, per:children)
    (32, 20, 17), # (per:parents, per:children, per:alternate_names)
    
    (17, 24, 24), # (per:alternate_names, per:countries_of_residence, per:countries_of_residence)
    (24, 17, 24), # (per:countries_of_residence, per:alternate_names, per:countries_of_residence)
    (24, 24, 17), # (per:countries_of_residence, per:countries_of_residence, per:alternate_names)

    (17, 21, 21), # (per:alternate_names, per:cities_of_residence, per:cities_of_residence)
    (21, 17, 21), # (per:cities_of_residence, per:alternate_names, per:cities_of_residence)
    (21, 21, 17), # (per:cities_of_residence, per:cities_of_residence, per:alternate_names)

    (17, 39, 39), # (per:alternate_names, per:stateorprovinces_of_residence, per:stateorprovinces_of_residence)
    (39, 17, 39), # (per:stateorprovinces_of_residence, per:alternate_names, per:stateorprovinces_of_residence)
    (39, 39, 17), # (per:stateorprovinces_of_residence, per:stateorprovinces_of_residence, per:alternate_names)

    (17, 25, 25), # (per:alternate_names, per:country_of_birth, per:country_of_birth)
    (25, 17, 25), # (per:country_of_birth, per:alternate_names, per:country_of_birth)
    (25, 25, 17), # (per:country_of_birth, per:country_of_birth, per:alternate_names)

    (17, 22, 22), # (per:alternate_names, per:city_of_birth, per:city_of_birth)
    (22, 17, 22), # (per:city_of_birth, per:alternate_names, per:city_of_birth)
    (22, 22, 17), # (per:city_of_birth, per:city_of_birth, per:alternate_names)

    (17, 37, 37), # (per:alternate_names, per:stateorprovince_of_birth, per:stateorprovince_of_birth)
    (37, 17, 37), # (per:stateorprovince_of_birth, per:alternate_names, per:stateorprovince_of_birth)
    (37, 37, 17), # (per:stateorprovince_of_birth, per:stateorprovince_of_birth, per:alternate_names)

    (17, 26, 26), # (per:alternate_names, per:country_of_death, per:country_of_death)
    (26, 17, 26), # (per:country_of_death, per:alternate_names, per:country_of_death)
    (26, 26, 17), # (per:country_of_death, per:country_of_death, per:alternate_names)

    (17, 23, 23), # (per:alternate_names, per:city_of_death, per:city_of_death)
    (23, 17, 23), # (per:city_of_death, per:alternate_names, per:city_of_death)
    (23, 23, 17), # (per:city_of_death, per:city_of_death, per:alternate_names)

    (17, 38, 38), # (per:alternate_names, per:stateorprovince_of_death, per:stateorprovince_of_death)
    (38, 17, 38), # (per:stateorprovince_of_death, per:alternate_names, per:stateorprovince_of_death)
    (38, 38, 17), # (per:stateorprovince_of_death, per:stateorprovince_of_death, per:alternate_names)

    (36, 20, 20), # (per:spouse, per:children, per:children)
    (20, 36, 20), # (per:children, per:spouse, per:children)
    (20, 20, 36), # (per:children, per:children, per:spouse)

    (36, 32, 32), # (per:spouse, per:parents, per:parents)
    (32, 36, 32), # (per:parents, per:spouse, per:parents)
    (32, 32, 36), # (per:parents, per:parents, per:spouse)

    (36, 20, 32), # (per:spouse, per:children, per:parents)
    (20, 36, 32), # (per:children, per:spouse, per:parents)
    (20, 32, 36), # (per:children, per:parents, per:spouse)

    (36, 32, 20), # (per:spouse, per:parents, per:children)
    (32, 36, 20), # (per:parents, per:spouse, per:children)
    (32, 20, 36), # (per:parents, per:children, per:spouse)
    
    (36, 24, 24), # (per:spouse, per:countries_of_residence, per:countries_of_residence)
    (24, 36, 24), # (per:countries_of_residence, per:spouse, per:countries_of_residence)
    (24, 24, 36), # (per:countries_of_residence, per:countries_of_residence, per:spouse)

    (36, 21, 21), # (per:spouse, per:cities_of_residence, per:cities_of_residence)
    (21, 36, 21), # (per:cities_of_residence, per:spouse, per:cities_of_residence)
    (21, 21, 36), # (per:cities_of_residence, per:cities_of_residence, per:spouse)

    (36, 39, 39), # (per:spouse, per:stateorprovinces_of_residence, per:stateorprovinces_of_residence)
    (39, 36, 39), # (per:stateorprovinces_of_residence, per:spouse, per:stateorprovinces_of_residence)
    (39, 39, 36), # (per:stateorprovinces_of_residence, per:stateorprovinces_of_residence, per:spouse)

    (36, 25, 25), # (per:spouse, per:country_of_birth, per:country_of_birth)
    (25, 36, 25), # (per:country_of_birth, per:spouse, per:country_of_birth)
    (25, 25, 36), # (per:country_of_birth, per:country_of_birth, per:spouse)

    (36, 22, 22), # (per:spouse, per:city_of_birth, per:city_of_birth)
    (22, 36, 22), # (per:city_of_birth, per:spouse, per:city_of_birth)
    (22, 22, 36), # (per:city_of_birth, per:city_of_birth, per:spouse)

    (36, 37, 37), # (per:spouse, per:stateorprovince_of_birth, per:stateorprovince_of_birth)
    (37, 36, 37), # (per:stateorprovince_of_birth, per:spouse, per:stateorprovince_of_birth)
    (37, 37, 36), # (per:stateorprovince_of_birth, per:stateorprovince_of_birth, per:spouse)

    (36, 26, 26), # (per:spouse, per:country_of_death, per:country_of_death)
    (26, 36, 26), # (per:country_of_death, per:spouse, per:country_of_death)
    (26, 26, 36), # (per:country_of_death, per:country_of_death, per:spouse)

    (36, 23, 23), # (per:spouse, per:city_of_death, per:city_of_death)
    (23, 36, 23), # (per:city_of_death, per:spouse, per:city_of_death)
    (23, 23, 36), # (per:city_of_death, per:city_of_death, per:spouse)

    (36, 38, 38), # (per:spouse, per:stateorprovince_of_death, per:stateorprovince_of_death)
    (38, 36, 38), # (per:stateorprovince_of_death, per:spouse, per:stateorprovince_of_death)
    (38, 38, 36), # (per:stateorprovince_of_death, per:stateorprovince_of_death, per:spouse)

    (35, 20, 20), # (per:siblings, per:children, per:children)
    (20, 35, 20), # (per:children, per:siblings, per:children)
    (20, 20, 35), # (per:children, per:children, per:siblings)

    (35, 32, 32), # (per:siblings, per:parents, per:parents)
    (32, 35, 32), # (per:parents, per:siblings, per:parents)
    (32, 32, 35), # (per:parents, per:parents, per:siblings)

    (35, 20, 32), # (per:siblings, per:children, per:parents)
    (20, 35, 32), # (per:children, per:siblings, per:parents)
    (20, 32, 35), # (per:children, per:parents, per:siblings)

    (35, 32, 20), # (per:siblings, per:parents, per:children)
    (32, 35, 20), # (per:parents, per:siblings, per:children)
    (32, 20, 35), # (per:parents, per:children, per:siblings)

    (35, 24, 24), # (per:siblings, per:countries_of_residence, per:countries_of_residence)
    (24, 35, 24), # (per:countries_of_residence, per:siblings, per:countries_of_residence)
    (24, 24, 35), # (per:countries_of_residence, per:countries_of_residence, per:siblings)

    (35, 21, 21), # (per:siblings, per:cities_of_residence, per:cities_of_residence)
    (21, 35, 21), # (per:cities_of_residence, per:siblings, per:cities_of_residence)
    (21, 21, 35), # (per:cities_of_residence, per:cities_of_residence, per:siblings)

    (35, 39, 39), # (per:siblings, per:stateorprovinces_of_residence, per:stateorprovinces_of_residence)
    (39, 35, 39), # (per:stateorprovinces_of_residence, per:siblings, per:stateorprovinces_of_residence)
    (39, 39, 35), # (per:stateorprovinces_of_residence, per:stateorprovinces_of_residence, per:siblings)    

    (35, 25, 25), # (per:siblings, per:country_of_birth, per:country_of_birth)
    (25, 35, 25), # (per:country_of_birth, per:siblings, per:country_of_birth)
    (25, 25, 35), # (per:country_of_birth, per:country_of_birth, per:siblings)

    (35, 22, 22), # (per:siblings, per:city_of_birth, per:city_of_birth)
    (22, 35, 22), # (per:city_of_birth, per:siblings, per:city_of_birth)
    (22, 22, 35), # (per:city_of_birth, per:city_of_birth, per:siblings)

    (35, 37, 37), # (per:siblings, per:stateorprovince_of_birth, per:stateorprovince_of_birth)
    (37, 35, 37), # (per:stateorprovince_of_birth, per:siblings, per:stateorprovince_of_birth)
    (37, 37, 35), # (per:stateorprovince_of_birth, per:stateorprovince_of_birth, per:siblings)

    (35, 26, 26), # (per:siblings, per:country_of_death, per:country_of_death)
    (26, 35, 26), # (per:country_of_death, per:siblings, per:country_of_death)
    (26, 26, 35), # (per:country_of_death, per:country_of_death, per:siblings)

    (35, 23, 23), # (per:siblings, per:city_of_death, per:city_of_death)
    (23, 35, 23), # (per:city_of_death, per:siblings, per:city_of_death)
    (23, 23, 35), # (per:city_of_death, per:city_of_death, per:siblings)

    (35, 38, 38), # (per:siblings, per:stateorprovince_of_death, per:stateorprovince_of_death)
    (38, 35, 38), # (per:stateorprovince_of_death, per:siblings, per:stateorprovince_of_death)
    (38, 38, 35), # (per:stateorprovince_of_death, per:stateorprovince_of_death, per:siblings)

    (32, 24, 24), # (per:parents, per:countries_of_residence, per:countries_of_residence)
    (24, 32, 24), # (per:countries_of_residence, per:parents, per:countries_of_residence)
    (24, 24, 32), # (per:countries_of_residence, per:countries_of_residence, per:parents)

    (32, 21, 21), # (per:parents, per:cities_of_residence, per:cities_of_residence)
    (21, 32, 21), # (per:cities_of_residence, per:parents, per:cities_of_residence)
    (21, 21, 32), # (per:cities_of_residence, per:cities_of_residence, per:parents)

    (32, 39, 39), # (per:parents, per:stateorprovinces_of_residence, per:stateorprovinces_of_residence)
    (39, 32, 39), # (per:stateorprovinces_of_residence, per:parents, per:stateorprovinces_of_residence)
    (39, 39, 32), # (per:stateorprovinces_of_residence, per:stateorprovinces_of_residence, per:parents)

    (32, 25, 25), # (per:parents, per:country_of_birth, per:country_of_birth)
    (25, 32, 25), # (per:country_of_birth, per:parents, per:country_of_birth)
    (25, 25, 32), # (per:country_of_birth, per:country_of_birth, per:parents)

    (32, 22, 22), # (per:parents, per:city_of_birth, per:city_of_birth)
    (22, 32, 22), # (per:city_of_birth, per:parents, per:city_of_birth)
    (22, 22, 32), # (per:city_of_birth, per:city_of_birth, per:parents)

    (32, 37, 37), # (per:parents, per:stateorprovince_of_birth, per:stateorprovince_of_birth)
    (37, 32, 37), # (per:stateorprovince_of_birth, per:parents, per:stateorprovince_of_birth)
    (37, 37, 32), # (per:stateorprovince_of_birth, per:stateorprovince_of_birth, per:parents)

    (32, 26, 26), # (per:parents, per:country_of_death, per:country_of_death)
    (26, 32, 26), # (per:country_of_death, per:parents, per:country_of_death)
    (26, 26, 32), # (per:country_of_death, per:country_of_death, per:parents)

    (32, 23, 23), # (per:parents, per:city_of_death, per:city_of_death)
    (23, 32, 23), # (per:city_of_death, per:parents, per:city_of_death)
    (23, 23, 32), # (per:city_of_death, per:city_of_death, per:parents)

    (32, 38, 38), # (per:parents, per:stateorprovince_of_death, per:stateorprovince_of_death)
    (38, 32, 38), # (per:stateorprovince_of_death, per:parents, per:stateorprovince_of_death)
    (38, 38, 32), # (per:stateorprovince_of_death, per:stateorprovince_of_death, per:parents)

    (20, 24, 24), # (per:children, per:countries_of_residence, per:countries_of_residence)
    (24, 20, 24), # (per:countries_of_residence, per:children, per:countries_of_residence)
    (24, 24, 20), # (per:countries_of_residence, per:countries_of_residence, per:children)

    (20, 21, 21), # (per:children, per:cities_of_residence, per:cities_of_residence)
    (21, 20, 21), # (per:cities_of_residence, per:children, per:cities_of_residence)
    (21, 21, 20), # (per:cities_of_residence, per:cities_of_residence, per:children)

    (20, 39, 39), # (per:children, per:stateorprovinces_of_residence, per:stateorprovinces_of_residence)
    (39, 20, 39), # (per:stateorprovinces_of_residence, per:children, per:stateorprovinces_of_residence)
    (39, 39, 20), # (per:stateorprovinces_of_residence, per:stateorprovinces_of_residence, per:children)

    (20, 25, 25), # (per:children, per:country_of_birth, per:country_of_birth)
    (25, 20, 25), # (per:country_of_birth, per:children, per:country_of_birth)
    (25, 25, 20), # (per:country_of_birth, per:country_of_birth, per:children)

    (20, 22, 22), # (per:children, per:city_of_birth, per:city_of_birth)
    (22, 20, 22), # (per:city_of_birth, per:children, per:city_of_birth)
    (22, 22, 20), # (per:city_of_birth, per:city_of_birth, per:children)

    (20, 37, 37), # (per:children, per:stateorprovince_of_birth, per:stateorprovince_of_birth)
    (37, 20, 37), # (per:stateorprovince_of_birth, per:children, per:stateorprovince_of_birth)
    (37, 37, 20), # (per:stateorprovince_of_birth, per:stateorprovince_of_birth, per:children)

    (20, 26, 26), # (per:children, per:country_of_death, per:country_of_death)
    (26, 20, 26), # (per:country_of_death, per:children, per:country_of_death)
    (26, 26, 20), # (per:country_of_death, per:country_of_death, per:children)

    (20, 23, 23), # (per:children, per:city_of_death, per:city_of_death)
    (23, 20, 23), # (per:city_of_death, per:children, per:city_of_death)
    (23, 23, 20), # (per:city_of_death, per:city_of_death, per:children)

    (20, 38, 38), # (per:children, per:stateorprovince_of_death, per:stateorprovince_of_death)
    (38, 20, 38), # (per:stateorprovince_of_death, per:children, per:stateorprovince_of_death)
    (38, 38, 20), # (per:stateorprovince_of_death, per:stateorprovince_of_death, per:children)

    (36, 33, 33), # (per:spouse, per:religion, per:religion)
    (33, 36, 33), # (per:religion, per:spouse, per:religion)
    (33, 33, 36), # (per:religion, per:religion, per:spouse)

    (35, 33, 33), # (per:siblings, per:religion, per:religion)
    (33, 35, 33), # (per:religion, per:siblings, per:religion)
    (33, 33, 35), # (per:religion, per:religion, per:siblings)

    (32, 33, 33), # (per:parents, per:religion, per:religion)
    (33, 32, 33), # (per:religion, per:parents, per:religion)
    (33, 33, 32), # (per:religion, per:religion, per:parents)

    (20, 33, 33), # (per:children, per:religion, per:religion)
    (33, 20, 33), # (per:religion, per:children, per:religion)
    (33, 33, 20), # (per:religion, per:religion, per:children)

    (36, 34, 34), # (per:spouse, per:schools_attended, per:schools_attended)
    (34, 36, 34), # (per:schools_attended, per:spouse, per:schools_attended)
    (34, 34, 36), # (per:schools_attended, per:schools_attended, per:spouse)

    (35, 34, 34), # (per:siblings, per:schools_attended, per:schools_attended)
    (34, 35, 34), # (per:schools_attended, per:siblings, per:schools_attended)
    (34, 34, 35), # (per:schools_attended, per:schools_attended, per:siblings)

    (32, 34, 34), # (per:parents, per:schools_attended, per:schools_attended)
    (34, 32, 34), # (per:schools_attended, per:parents, per:schools_attended)
    (34, 34, 32), # (per:schools_attended, per:schools_attended, per:parents)

    (20, 34, 34), # (per:children, per:schools_attended, per:schools_attended)
    (34, 20, 34), # (per:schools_attended, per:children, per:schools_attended)
    (34, 34, 20), # (per:schools_attended, per:schools_attended, per:children)

    (36, 30, 30), # (per:spouse, per:origin, per:origin)
    (30, 36, 30), # (per:origin, per:spouse, per:origin)
    (30, 30, 36), # (per:origin, per:origin, per:spouse)

    (35, 30, 30), # (per:siblings, per:origin, per:origin)
    (30, 35, 30), # (per:origin, per:siblings, per:origin)
    (30, 30, 35), # (per:origin, per:origin, per:siblings)

    (32, 30, 30), # (per:parents, per:origin, per:origin)
    (30, 32, 30), # (per:origin, per:parents, per:origin)
    (30, 30, 32), # (per:origin, per:origin, per:parents)

    (20, 30, 30), # (per:children, per:origin, per:origin)
    (30, 20, 30), # (per:origin, per:children, per:origin)
    (30, 30, 20), # (per:origin, per:origin, per:children)

    (6, 1, 1), # (org:member_of, org:city_of_headquarters, org:city_of_headquarters)
    (1, 6, 1), # (org:city_of_headquarters, org:member_of, org:city_of_headquarters)
    (1, 1, 6), # (org:city_of_headquarters, org:city_of_headquarters, org:member_of)

    (6, 2, 2), # (org:member_of, org:country_of_headquarters, org:country_of_headquarters)
    (2, 6, 2), # (org:country_of_headquarters, org:member_of, org:country_of_headquarters)
    (2, 2, 6), # (org:country_of_headquarters, org:country_of_headquarters, org:member_of)

    (6, 12, 12), # (org:member_of, org:stateorprovince_of_headquarters, org:stateorprovince_of_headquarters)
    (12, 6, 12), # (org:stateorprovince_of_headquarters, org:member_of, org:stateorprovince_of_headquarters)
    (12, 12, 6), # (org:stateorprovince_of_headquarters, org:stateorprovince_of_headquarters, org:member_of)

    (6, 10, 10), # (org:member_of, org:political/religious_affiliation, org:political/religious_affiliation)
    (10, 6, 10), # (org:political/religious_affiliation, org:member_of, org:political/religious_affiliation)
    (10, 10, 6), # (org:political/religious_affiliation, org:political/religious_affiliation, org:member_of)

    (6, 5, 5), # (org:member_of, org:founded_by, org:founded_by)
    (5, 6, 5), # (org:founded_by, org:member_of, org:founded_by)
    (5, 5, 6), # (org:founded_by, org:founded_by, org:member_of)

    (6, 14, 14), # (org:member_of, org:top_members/employees, org:top_members/employees)
    (14, 6, 14), # (org:top_members/employees, org:member_of, org:top_members/employees)
    (14, 14, 6), # (org:top_members/employees, org:top_members/employees, org:member_of)

    (6, 29, 29), # (org:member_of, per:employee_of, per:employee_of)
    (29, 6, 29), # (per:employee_of, org:member_of, per:employee_of)
    (29, 29, 6), # (per:employee_of, per:employee_of, org:member_of)

    (6, 5, 14), # (org:member_of, org:founded_by, org:top_members/employees)
    (5, 6, 14), # (org:founded_by, org:member_of, org:top_members/employees)
    (5, 14, 6), # (org:founded_by, org:top_members/employees, org:member_of)

    (6, 5, 29), # (org:member_of, org:founded_by, per:employee_of)
    (5, 6, 29), # (org:founded_by, org:member_of, per:employee_of)
    (5, 29, 6), # (org:founded_by, per:employee_of, org:member_of)

    (6, 14, 5), # (org:member_of, org:top_members/employees, org:founded_by)
    (14, 6, 5), # (org:top_members/employees, org:member_of, org:founded_by)
    (14, 5, 6), # (org:top_members/employees, org:founded_by, org:member_of)

    (6, 14, 29), # (org:member_of, org:top_members/employees, per:employee_of)
    (14, 6, 29), # (org:top_members/employees, org:member_of, per:employee_of)
    (14, 29, 6), # (org:top_members/employees, per:employee_of, org:member_of)

    (6, 29, 5), # (org:member_of, per:employee_of, org:founded_by)
    (29, 6, 5), # (per:employee_of, org:member_of, org:founded_by)
    (29, 5, 6), # (per:employee_of, org:founded_by, org:member_of)

    (6, 29, 14), # (org:member_of, per:employee_of, org:top_members/employees)
    (29, 6, 14), # (per:employee_of, org:member_of, org:top_members/employees)
    (29, 14, 6), # (per:employee_of, org:top_members/employees, org:member_of)

    (7, 1, 1), # (org:members, org:city_of_headquarters, org:city_of_headquarters)
    (1, 7, 1), # (org:city_of_headquarters, org:members, org:city_of_headquarters)
    (1, 1, 7), # (org:city_of_headquarters, org:city_of_headquarters, org:members)

    (7, 2, 2), # (org:members, org:country_of_headquarters, org:country_of_headquarters)
    (2, 7, 2), # (org:country_of_headquarters, org:members, org:country_of_headquarters)
    (2, 2, 7), # (org:country_of_headquarters, org:country_of_headquarters, org:members)

    (7, 12, 12), # (org:members, org:stateorprovince_of_headquarters, org:stateorprovince_of_headquarters)
    (12, 7, 12), # (org:stateorprovince_of_headquarters, org:members, org:stateorprovince_of_headquarters)
    (12, 12, 7), # (org:stateorprovince_of_headquarters, org:stateorprovince_of_headquarters, org:members)

    (7, 10, 10), # (org:members, org:political/religious_affiliation, org:political/religious_affiliation)
    (10, 7, 10), # (org:political/religious_affiliation, org:members, org:political/religious_affiliation)
    (10, 10, 7), # (org:political/religious_affiliation, org:political/religious_affiliation, org:members)

    (7, 5, 5), # (org:members, org:founded_by, org:founded_by)
    (5, 7, 5), # (org:founded_by, org:members, org:founded_by)
    (5, 5, 7), # (org:founded_by, org:founded_by, org:members)

    (7, 14, 14), # (org:members, org:top_members/employees, org:top_members/employees)
    (14, 7, 14), # (org:top_members/employees, org:members, org:top_members/employees)
    (14, 14, 7), # (org:top_members/employees, org:top_members/employees, org:members)

    (7, 29, 29), # (org:members, per:employee_of, per:employee_of)
    (29, 7, 29), # (per:employee_of, org:members, per:employee_of)
    (29, 29, 7), # (per:employee_of, per:employee_of, org:members)

    (7, 5, 14), # (org:members, org:founded_by, org:top_members/employees)
    (5, 7, 14), # (org:founded_by, org:members, org:top_members/employees)
    (5, 14, 7), # (org:founded_by, org:top_members/employees, org:members)

    (7, 5, 29), # (org:members, org:founded_by, per:employee_of)
    (5, 7, 29), # (org:founded_by, org:members, per:employee_of)
    (5, 29, 7), # (org:founded_by, per:employee_of, org:members)

    (7, 14, 5), # (org:members, org:top_members/employees, org:founded_by)
    (14, 7, 5), # (org:top_members/employees, org:members, org:founded_by)
    (14, 5, 7), # (org:top_members/employees, org:founded_by, org:members)

    (7, 14, 29), # (org:members, org:top_members/employees, per:employee_of)
    (14, 7, 29), # (org:top_members/employees, org:members, per:employee_of)
    (14, 29, 7), # (org:top_members/employees, per:employee_of, org:members)

    (7, 29, 5), # (org:members, per:employee_of, org:founded_by)
    (29, 7, 5), # (per:employee_of, org:members, org:founded_by)
    (29, 5, 7), # (per:employee_of, org:founded_by, org:members)

    (7, 29, 14), # (org:members, per:employee_of, org:top_members/employees)
    (29, 7, 14), # (per:employee_of, org:members, org:top_members/employees)
    (29, 14, 7), # (per:employee_of, org:top_members/employees, org:members)

    (9, 1, 1), # (org:parents, org:city_of_headquarters, org:city_of_headquarters)
    (1, 9, 1), # (org:city_of_headquarters, org:parents, org:city_of_headquarters)
    (1, 1, 9), # (org:city_of_headquarters, org:city_of_headquarters, org:parents)

    (9, 2, 2), # (org:parents, org:country_of_headquarters, org:country_of_headquarters)
    (2, 9, 2), # (org:country_of_headquarters, org:parents, org:country_of_headquarters)
    (2, 2, 9), # (org:country_of_headquarters, org:country_of_headquarters, org:parents)

    (9, 12, 12), # (org:parents, org:stateorprovince_of_headquarters, org:stateorprovince_of_headquarters)
    (12, 9, 12), # (org:stateorprovince_of_headquarters, org:parents, org:stateorprovince_of_headquarters)
    (12, 12, 9), # (org:stateorprovince_of_headquarters, org:stateorprovince_of_headquarters, org:parents)

    (9, 10, 10), # (org:parents, org:political/religious_affiliation, org:political/religious_affiliation)
    (10, 9, 10), # (org:political/religious_affiliation, org:parents, org:political/religious_affiliation)
    (10, 10, 9), # (org:political/religious_affiliation, org:political/religious_affiliation, org:parents)

    (9, 5, 5), # (org:parents, org:founded_by, org:founded_by)
    (5, 9, 5), # (org:founded_by, org:parents, org:founded_by)
    (5, 5, 9), # (org:founded_by, org:founded_by, org:parents)

    (9, 14, 14), # (org:parents, org:top_members/employees, org:top_members/employees)
    (14, 9, 14), # (org:top_members/employees, org:parents, org:top_members/employees)
    (14, 14, 9), # (org:top_members/employees, org:top_members/employees, org:parents)

    (9, 29, 29), # (org:parents, per:employee_of, per:employee_of)
    (29, 9, 29), # (per:employee_of, org:parents, per:employee_of)
    (29, 29, 9), # (per:employee_of, per:employee_of, org:parents)

    (9, 5, 14), # (org:parents, org:founded_by, org:top_members/employees)
    (5, 9, 14), # (org:founded_by, org:parents, org:top_members/employees)
    (5, 14, 9), # (org:founded_by, org:top_members/employees, org:parents)

    (9, 5, 29), # (org:parents, org:founded_by, per:employee_of)
    (5, 9, 29), # (org:founded_by, org:parents, per:employee_of)
    (5, 29, 9), # (org:founded_by, per:employee_of, org:parents)

    (9, 14, 5), # (org:parents, org:top_members/employees, org:founded_by)
    (14, 9, 5), # (org:top_members/employees, org:parents, org:founded_by)
    (14, 5, 9), # (org:top_members/employees, org:founded_by, org:parents)

    (9, 14, 29), # (org:parents, org:top_members/employees, per:employee_of)
    (14, 9, 29), # (org:top_members/employees, org:parents, per:employee_of)
    (14, 29, 9), # (org:top_members/employees, per:employee_of, org:parents)

    (9, 29, 5), # (org:parents, per:employee_of, org:founded_by)
    (29, 9, 5), # (per:employee_of, org:parents, org:founded_by)
    (29, 5, 9), # (per:employee_of, org:founded_by, org:parents)

    (9, 29, 14), # (org:parents, per:employee_of, org:top_members/employees)
    (29, 9, 14), # (per:employee_of, org:parents, org:top_members/employees)
    (29, 14, 9), # (per:employee_of, org:top_members/employees, org:parents)

    (11, 1, 1), # (org:shareholders, org:city_of_headquarters, org:city_of_headquarters)
    (1, 11, 1), # (org:city_of_headquarters, org:shareholders, org:city_of_headquarters)
    (1, 1, 11), # (org:city_of_headquarters, org:city_of_headquarters, org:shareholders)

    (11, 2, 2), # (org:shareholders, org:country_of_headquarters, org:country_of_headquarters)
    (2, 11, 2), # (org:country_of_headquarters, org:shareholders, org:country_of_headquarters)
    (2, 2, 11), # (org:country_of_headquarters, org:country_of_headquarters, org:shareholders)

    (11, 12, 12), # (org:shareholders, org:stateorprovince_of_headquarters, org:stateorprovince_of_headquarters)
    (12, 11, 12), # (org:stateorprovince_of_headquarters, org:shareholders, org:stateorprovince_of_headquarters)
    (12, 12, 11), # (org:stateorprovince_of_headquarters, org:stateorprovince_of_headquarters, org:shareholders)

    (11, 10, 10), # (org:shareholders, org:political/religious_affiliation, org:political/religious_affiliation)
    (10, 11, 10), # (org:political/religious_affiliation, org:shareholders, org:political/religious_affiliation)
    (10, 10, 11), # (org:political/religious_affiliation, org:political/religious_affiliation, org:shareholders)

    (11, 5, 5), # (org:shareholders, org:founded_by, org:founded_by)
    (5, 11, 5), # (org:founded_by, org:shareholders, org:founded_by)
    (5, 5, 11), # (org:founded_by, org:founded_by, org:shareholders)

    (11, 14, 14), # (org:shareholders, org:top_members/employees, org:top_members/employees)
    (14, 11, 14), # (org:top_members/employees, org:shareholders, org:top_members/employees)
    (14, 14, 11), # (org:top_members/employees, org:top_members/employees, org:shareholders)

    (11, 29, 29), # (org:shareholders, per:employee_of, per:employee_of)
    (29, 11, 29), # (per:employee_of, org:shareholders, per:employee_of)
    (29, 29, 11), # (per:employee_of, per:employee_of, org:shareholders)

    (11, 5, 14), # (org:shareholders, org:founded_by, org:top_members/employees)
    (5, 11, 14), # (org:founded_by, org:shareholders, org:top_members/employees)
    (5, 14, 11), # (org:founded_by, org:top_members/employees, org:shareholders)

    (11, 5, 29), # (org:shareholders, org:founded_by, per:employee_of)
    (5, 11, 29), # (org:founded_by, org:shareholders, per:employee_of)
    (5, 29, 11), # (org:founded_by, per:employee_of, org:shareholders)

    (11, 14, 5), # (org:shareholders, org:top_members/employees, org:founded_by)
    (14, 11, 5), # (org:top_members/employees, org:shareholders, org:founded_by)
    (14, 5, 11), # (org:top_members/employees, org:founded_by, org:shareholders)

    (11, 14, 29), # (org:shareholders, org:top_members/employees, per:employee_of)
    (14, 11, 29), # (org:top_members/employees, org:shareholders, per:employee_of)
    (14, 29, 11), # (org:top_members/employees, per:employee_of, org:shareholders)

    (11, 29, 5), # (org:shareholders, per:employee_of, org:founded_by)
    (29, 11, 5), # (per:employee_of, org:shareholders, org:founded_by)
    (29, 5, 11), # (per:employee_of, org:founded_by, org:shareholders)

    (11, 29, 14), # (org:shareholders, per:employee_of, org:top_members/employees)
    (29, 11, 14), # (per:employee_of, org:shareholders, org:top_members/employees)
    (29, 14, 11), # (per:employee_of, org:top_members/employees, org:shareholders)

    (13, 1, 1), # (org:subsidiaries, org:city_of_headquarters, org:city_of_headquarters)
    (1, 13, 1), # (org:city_of_headquarters, org:subsidiaries, org:city_of_headquarters)
    (1, 1, 13), # (org:city_of_headquarters, org:city_of_headquarters, org:subsidiaries)

    (13, 2, 2), # (org:subsidiaries, org:country_of_headquarters, org:country_of_headquarters)
    (2, 13, 2), # (org:country_of_headquarters, org:subsidiaries, org:country_of_headquarters)
    (2, 2, 13), # (org:country_of_headquarters, org:country_of_headquarters, org:subsidiaries)

    (13, 12, 12), # (org:subsidiaries, org:stateorprovince_of_headquarters, org:stateorprovince_of_headquarters)
    (12, 13, 12), # (org:stateorprovince_of_headquarters, org:subsidiaries, org:stateorprovince_of_headquarters)
    (12, 12, 13), # (org:stateorprovince_of_headquarters, org:stateorprovince_of_headquarters, org:subsidiaries)

    (13, 10, 10), # (org:subsidiaries, org:political/religious_affiliation, org:political/religious_affiliation)
    (10, 13, 10), # (org:political/religious_affiliation, org:subsidiaries, org:political/religious_affiliation)
    (10, 10, 13), # (org:political/religious_affiliation, org:political/religious_affiliation, org:subsidiaries)

    (13, 5, 5), # (org:subsidiaries, org:founded_by, org:founded_by)
    (5, 13, 5), # (org:founded_by, org:subsidiaries, org:founded_by)
    (5, 5, 13), # (org:founded_by, org:founded_by, org:subsidiaries)

    (13, 14, 14), # (org:subsidiaries, org:top_members/employees, org:top_members/employees)
    (14, 13, 14), # (org:top_members/employees, org:subsidiaries, org:top_members/employees)
    (14, 14, 13), # (org:top_members/employees, org:top_members/employees, org:subsidiaries)

    (13, 29, 29), # (org:subsidiaries, per:employee_of, per:employee_of)
    (29, 13, 29), # (per:employee_of, org:subsidiaries, per:employee_of)
    (29, 29, 13), # (per:employee_of, per:employee_of, org:subsidiaries)

    (13, 5, 14), # (org:subsidiaries, org:founded_by, org:top_members/employees)
    (5, 13, 14), # (org:founded_by, org:subsidiaries, org:top_members/employees)
    (5, 14, 13), # (org:founded_by, org:top_members/employees, org:subsidiaries)

    (13, 5, 29), # (org:subsidiaries, org:founded_by, per:employee_of)
    (5, 13, 29), # (org:founded_by, org:subsidiaries, per:employee_of)
    (5, 29, 13), # (org:founded_by, per:employee_of, org:subsidiaries)

    (13, 14, 5), # (org:subsidiaries, org:top_members/employees, org:founded_by)
    (14, 13, 5), # (org:top_members/employees, org:subsidiaries, org:founded_by)
    (14, 5, 13), # (org:top_members/employees, org:founded_by, org:subsidiaries)

    (13, 14, 29), # (org:subsidiaries, org:top_members/employees, per:employee_of)
    (14, 13, 29), # (org:top_members/employees, org:subsidiaries, per:employee_of)
    (14, 29, 13), # (org:top_members/employees, per:employee_of, org:subsidiaries)

    (13, 29, 5), # (org:subsidiaries, per:employee_of, org:founded_by)
    (29, 13, 5), # (per:employee_of, org:subsidiaries, org:founded_by)
    (29, 5, 13), # (per:employee_of, org:founded_by, org:subsidiaries)

    (13, 29, 14), # (org:subsidiaries, per:employee_of, org:top_members/employees)
    (29, 13, 14), # (per:employee_of, org:subsidiaries, org:top_members/employees)
    (29, 14, 13), # (per:employee_of, org:top_members/employees, org:subsidiaries)

    (0, 1, 1), # (org:alternate_names, org:city_of_headquarters, org:city_of_headquarters)
    (1, 0, 1), # (org:city_of_headquarters, org:alternate_names, org:city_of_headquarters)
    (1, 1, 0), # (org:city_of_headquarters, org:city_of_headquarters, org:alternate_names)

    (0, 2, 2), # (org:alternate_names, org:country_of_headquarters, org:country_of_headquarters)
    (2, 0, 2), # (org:country_of_headquarters, org:alternate_names, org:country_of_headquarters)
    (2, 2, 0), # (org:country_of_headquarters, org:country_of_headquarters, org:alternate_names)

    (0, 12, 12), # (org:alternate_names, org:stateorprovince_of_headquarters, org:stateorprovince_of_headquarters)
    (12, 0, 12), # (org:stateorprovince_of_headquarters, org:alternate_names, org:stateorprovince_of_headquarters)
    (12, 12, 0), # (org:stateorprovince_of_headquarters, org:stateorprovince_of_headquarters, org:alternate_names)

    (0, 10, 10), # (org:alternate_names, org:political/religious_affiliation, org:political/religious_affiliation)
    (10, 0, 10), # (org:political/religious_affiliation, org:alternate_names, org:political/religious_affiliation)
    (10, 10, 0), # (org:political/religious_affiliation, org:political/religious_affiliation, org:alternate_names)

    (0, 5, 5), # (org:alternate_names, org:founded_by, org:founded_by)
    (5, 0, 5), # (org:founded_by, org:alternate_names, org:founded_by)
    (5, 5, 0), # (org:founded_by, org:founded_by, org:alternate_names)

    (0, 14, 14), # (org:alternate_names, org:top_members/employees, org:top_members/employees)
    (14, 0, 14), # (org:top_members/employees, org:alternate_names, org:top_members/employees)
    (14, 14, 0), # (org:top_members/employees, org:top_members/employees, org:alternate_names)

    (0, 29, 29), # (org:alternate_names, per:employee_of, per:employee_of)
    (29, 0, 29), # (per:employee_of, org:alternate_names, per:employee_of)
    (29, 29, 0), # (per:employee_of, per:employee_of, org:alternate_names)

    (0, 5, 14), # (org:alternate_names, org:founded_by, org:top_members/employees)
    (5, 0, 14), # (org:founded_by, org:alternate_names, org:top_members/employees)
    (5, 14, 0), # (org:founded_by, org:top_members/employees, org:alternate_names)

    (0, 5, 29), # (org:alternate_names, org:founded_by, per:employee_of)
    (5, 0, 29), # (org:founded_by, org:alternate_names, per:employee_of)
    (5, 29, 0), # (org:founded_by, per:employee_of, org:alternate_names)

    (0, 14, 5), # (org:alternate_names, org:top_members/employees, org:founded_by)
    (14, 0, 5), # (org:top_members/employees, org:alternate_names, org:founded_by)
    (14, 5, 0), # (org:top_members/employees, org:founded_by, org:alternate_names)

    (0, 14, 29), # (org:alternate_names, org:top_members/employees, per:employee_of)
    (14, 0, 29), # (org:top_members/employees, org:alternate_names, per:employee_of)
    (14, 29, 0), # (org:top_members/employees, per:employee_of, org:alternate_names)

    (0, 29, 5), # (org:alternate_names, per:employee_of, org:founded_by)
    (29, 0, 5), # (per:employee_of, org:alternate_names, org:founded_by)
    (29, 5, 0), # (per:employee_of, org:founded_by, org:alternate_names)

    (0, 29, 14), # (org:alternate_names, per:employee_of, org:top_members/employees)
    (29, 0, 14), # (per:employee_of, org:alternate_names, org:top_members/employees)
    (29, 14, 0), # (per:employee_of, org:top_members/employees, org:alternate_names)
    

    



    


]