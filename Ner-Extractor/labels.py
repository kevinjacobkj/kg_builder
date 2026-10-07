LABELS = {
    # People
    "person": "PER",
    "job title": "TITLE",

    # Organizations
    "organization": "ORG",
    "company": "COMP",
    "department": "DEPT",
    "team": "TEAM",
    "business unit": "BU",

    # Products / technology
    "product": "PROD",
    "software": "SOFT",
    "platform": "PLAT",
    "technology": "TECH",
    "service": "SERV",
    "application": "APP",
    "database": "DB",

    # Locations
    "location": "LOC",
    "city": "CITY",
    "state or province": "STATE",
    "country": "COUNTRY",
    "region": "REGION",

    # Business / organizational concepts
    "project": "PROJECT",
    "program": "PROGRAM",
    "initiative": "INITIATIVE",
    "contract": "CONTRACT",
    "agreement": "AGREEMENT",

    # Business functions
    "department function": "FUNCTION",
    "business process": "PROCESS",

    # Other useful entities
    "event": "EVENT",
    "date": "DATE",
    "money": "MONEY",
    "percentage": "PERCENT",
    "email address": "EMAIL",
    "phone number": "PHONE",
    "url": "URL",
}

_PERSON = {"PER"}
_COMPANY = {"COMP", "ORG"}
_GROUP = {"DEPT", "TEAM", "BU"}
_PLACE = {"LOC", "CITY", "STATE", "COUNTRY", "REGION"}
_PRODUCT = {"PROD", "SOFT", "PLAT", "TECH", "SERV", "APP", "DB"}
_WORK = {"PROJECT", "PROGRAM", "INITIATIVE"}
_ACTIVITY = {"FUNCTION", "PROCESS"}
_DATE = {"DATE"}

# gliner-relex relation prompts -> (allowed head types, allowed tail types); other type pairs are dropped.
RELATIONS = {
    # People and roles
    "works at": (_PERSON, _COMPANY),
    "joined": (_PERSON, _COMPANY | _GROUP),
    "joined in": (_PERSON, _DATE),
    "has job title": (_PERSON, {"TITLE"}),
    "promoted to": (_PERSON, {"TITLE"}),
    "promoted in": (_PERSON, _DATE),
    "appointed": (_COMPANY, _PERSON),
    "appointed in": (_PERSON, _DATE),
    "founded": (_PERSON, _COMPANY),

    # Leadership and reporting lines
    "leads": (_PERSON, _GROUP | _WORK | _ACTIVITY),
    "oversees": (_PERSON, _GROUP | _WORK | _ACTIVITY | _PRODUCT),
    "manages": (_PERSON | _GROUP, _GROUP | _WORK | _ACTIVITY | _PRODUCT),
    "reports to": (_PERSON, _PERSON),
    "works with": (_PERSON | _GROUP, _PERSON | _GROUP),
    "part of": (_GROUP, _COMPANY | _GROUP),

    # Locations
    "headquartered in": (_COMPANY, _PLACE),
    "has office in": (_COMPANY, _PLACE),
    "based in": (_PERSON | _GROUP, _PLACE),
    "has customers in": (_COMPANY, _PLACE),

    # Companies, products and projects
    "acquired": (_COMPANY, _COMPANY),
    "was acquired in": (_COMPANY, _DATE),
    "launched": (_COMPANY | _GROUP, _PRODUCT | _WORK),
    "launched in": (_PRODUCT | _WORK, _DATE),
    "developed by": (_PRODUCT | _WORK, _COMPANY | _GROUP | _PERSON),
    "supports": (_GROUP, _PRODUCT | _WORK | _ACTIVITY),
    "sponsored": (_PERSON, _WORK | _PRODUCT),
    "implemented in": (_PRODUCT | _WORK | _ACTIVITY, _DATE),
}