# Amsterdam Museum Knowledge Graph — Navigation Guide

> **Purpose:** This document tells you exactly how the Amsterdam Museum (AHM) Knowledge Graph (KG) is structured and how to query it efficiently using the `tfm-kg` MCP tools. Read it before answering any question about the museum's collection, artists, or objects.

---

## 1. Available MCP Tools

| Tool | When to use |
|------|-------------|
| `tfm-kg:schema_summary` | Get a structural overview (top classes, predicates, example triples). Use once at the start of a session if you need orientation. |
| `tfm-kg:entity_search` | Full-text search across label-like predicates. Useful but returns limited results; prefer SPARQL for precision. |
| `tfm-kg:entity_facts` | Fetch all one-hop triples for a **known URI**. Best for drilling into a specific record after you have its URI. Set `include_incoming: true` to also see which nodes point *to* it. |
| `tfm-kg:sparql_query` | The primary workhorse. Write SPARQL SELECT queries for any search, filter, join, or aggregation. Always prefer this over browsing randomly. |
| `tfm-kg:ping` | Health check only. |

---

## 2. Base Namespace & URI Patterns

All Amsterdam Museum URIs share the base namespace:

```
http://purl.org/collections/nl/am/
```

Shorthand: `am:` in this document.

Key URI patterns (the number is the internal `priref`):

| Entity type | URI pattern | Example |
|-------------|-------------|---------|
| Artwork proxy (main record) | `am:proxy-{N}` | `am:proxy-1` |
| Physical thing (EDM) | `am:physical-{N}` | `am:physical-1` |
| Aggregation (EDM) | `am:aggregation-{N}` | `am:aggregation-1` |
| Person / artist | `am:p-{N}` | `am:p-31377` |
| Thesaurus concept / term | `am:t-{N}` | `am:t-12936` |

---

## 3. The Object Model — How an Artwork is Structured

Each museum object is represented as **three linked nodes** following the Europeana Data Model (EDM):

```
am:physical-N       ← the abstract "physical thing" (EDM class)
    ↑ aggregatedCHO
am:aggregation-N    ← holds media links (landing page, image URL)
    ↑ proxyIn
am:proxy-N          ← **the main descriptive record** (title, maker, date, etc.)
```

**In practice, always query `am:proxy-N`** — it carries all catalogue metadata. The `physical` and `aggregation` nodes are thin wrappers.

### 3.1 Artwork (Proxy) — Key Predicates

All predicates are in the `am:` namespace unless noted.

| Predicate | Value type | Notes |
|-----------|-----------|-------|
| `am:priref` | literal integer | Internal catalogue ID |
| `am:objectNumber` | literal string | Museum accession number, e.g. `"A 8"`, `"SA 7359"` |
| `am:title` | literal (lang: `nl` or `en`) | Object title; one object can have multiple titles |
| `am:productionDateStart` | literal string (year/date) | Earliest production date |
| `am:productionDateEnd` | literal string (year/date) | Latest production date |
| `am:productionPeriod` | literal string | Free-text period label |
| `am:productionPlace` | literal string | Place of production |
| `am:objectName` | URI → `am:t-{N}` (skos:Concept) | Object type (e.g. "prent", "schilderij") |
| `am:objectCategory` | URI → `am:t-{N}` | Broader category |
| `am:collection` | URI → `am:t-{N}` | Collection subdivision |
| `am:material` | URI → `am:t-{N}` | Material(s) used |
| `am:technique` | URI → `am:t-{N}` | Technique(s) used |
| `am:contentSubject` | URI → `am:t-{N}` | Subject/iconography terms |
| `am:contentMotifGeneral` | URI → `am:t-{N}` | General motif terms |
| `am:contentPersonName` | URI → `am:p-{N}` | Persons depicted |
| `am:associationPerson` | URI → `am:p-{N}` | Persons associated with the object |
| `am:associationSubject` | URI → `am:t-{N}` | Subject associations |
| `am:acquisitionDate` | literal string | Date of acquisition |
| `am:acquisitionMethod` | URI → `am:t-{N}` | How it was acquired |
| `am:creditLine` | literal string | Donor/provenance credit line |
| `am:ownershipHistoryFreeText` | literal string | Provenance notes |
| `am:relatedObjectReference` | URI → `am:proxy-{N}` | Links to related artworks |
| `am:partOfReference` | URI → `am:proxy-{N}` | Parent object (if part of a set) |
| `am:partsReference` | URI → `am:proxy-{N}` | Child objects |
| `am:documentation` | blank node → Documentation | Bibliographic references |
| `am:exhibition` | blank node → Exhibition | Exhibition history |
| `am:alternativenumber` | blank node → Alternativenumber | Alternative inventory numbers |
| `am:ahmteksten` | blank node → Ahmteksten | Internal text notes |
| `ore:proxyFor` | URI → `am:physical-{N}` | Link to physical node |
| `ore:proxyIn` | URI → `am:aggregation-{N}` | Link to aggregation node |

### 3.2 Maker (blank node)

The `am:maker` predicate on a proxy points to an **anonymous blank node** of type `am:Maker`. To get the person URI, traverse through it:

```sparql
?proxy am:maker ?makerBnode .
?makerBnode rdf:value ?personURI .         # → am:p-{N}
OPTIONAL { ?makerBnode am:creatorQualifier ?qualifier . }  # e.g. "naar" (after), "toegeschreven aan"
```

### 3.3 Dimensions (blank node)

The `am:dimension` predicate points to blank nodes of type `am:Dimension`:

```sparql
?proxy am:dimension ?dim .
?dim rdfs:label ?label .             # human-readable, e.g. "hoogte a 14.6 cm"
?dim am:dimensionType ?typeURI .     # → am:t-{N} (hoogte, breedte, etc.)
?dim am:dimensionUnit ?unit .        # e.g. "cm"
?dim am:dimensionValue ?value .      # numeric value as string
```

### 3.4 Location (blank node)

The `am:locat` predicate points to blank nodes of type `am:Locat` (location history entries):

```sparql
?proxy am:locat ?loc .
?loc rdfs:label ?label .
?loc am:currentLocation ?locationURI .           # → am:t-{N}
?loc am:currentLocationDateStart ?dateStart .
?loc am:currentLocationDateEnd ?dateEnd .        # absent = current location
?loc am:currentLocationNotes ?notes .
?loc am:currentLocationType ?type .              # "intern", "bruikleen", etc.
```

The **most recent** location (no `currentLocationDateEnd`) is the current one.

### 3.5 Reproduction (blank node)

The `am:reproduction` predicate points to blank nodes of type `am:Reproduction` (image records):

```sparql
?proxy am:reproduction ?rep .
?rep am:reproductionIdentifierURL ?path .   # file path (internal)
?rep am:reproductionReference ?filename .   # e.g. "A_11877_000.jpg"
?rep am:reproductionType ?typeURI .
?rep am:reproductionDate ?date .
?rep am:reproductionCreator ?personURI .
```

The **web-accessible image** URL is stored on the aggregation node:
```sparql
am:aggregation-N  edm:object  <image_url>
am:aggregation-N  edm:landingPage  <catalogue_page_url>
```

---

## 4. Person / Artist Records

Person URIs follow the pattern `am:p-{N}`.

| Predicate | Notes |
|-----------|-------|
| `am:priref` | Internal ID |
| `am:name` | Full name, format "Surname, Firstname" |
| `am:biography` | Free-text biographical note (in Dutch) |
| `am:birthDateStart` / `am:birthDateEnd` | Birth date (range) |
| `am:birthPlace` | Birth city/country |
| `am:birthDatePrecision` | Precision qualifier |
| `am:birthNotes` | Birth notes |
| `am:deathDateStart` / `am:deathDateEnd` | Death date (range) |
| `am:deathPlace` | Death city/country |
| `am:deathNotes` | Death notes |
| `am:nationality` | Nationality term |
| `am:occupation` | Occupation term URI |
| `am:equivalentName` | Variant/alternative names |
| `am:use` / `am:usedFor` | Authority control notes |
| `am:wasPresentAt` | Links to exhibitions |
| `am:selected` | Boolean string ("True"/"False") — flag for featured persons |

---

## 5. Thesaurus / Controlled Vocabulary Terms

Terms used in `am:material`, `am:technique`, `am:objectName`, `am:contentSubject`, etc. are all `skos:Concept` nodes with URI `am:t-{N}`.

To resolve a term URI to its human-readable label:

```sparql
SELECT ?label WHERE {
  am:t-12936  skos:prefLabel  ?label .
}
# Returns: "prent" (print/engraving)
```

The SKOS hierarchy is also present:
```sparql
am:t-{N}  skos:broader  am:t-{M}   # parent concept
am:t-{N}  skos:narrower  am:t-{M}  # child concept
am:t-{N}  am:termType   am:t-termtype{CODE}  # term category
```

### Key Term Type Codes

| URI suffix | Meaning |
|------------|---------|
| `t-termtypeOBJECT` | Object name (e.g. prent, schilderij) |
| `t-termtypeTECHN` | Technique (e.g. geëtst, olieverf) |
| `t-termtypeMATER` | Material |
| `t-termtypeSUBJECT` | Subject/iconography |
| `t-termtypeMOTIF` | Motif |
| `t-termtypeLOCAT` | Location/place (storage) |
| `t-termtypeGEOKEYW` | Geographic keyword |
| `t-termtypePLACE` | Place name |
| `t-termtypeOBJCAT` | Object category |
| `t-termtypeCLASS` | Classification |
| `t-termtypePERIOD` | Historical period |
| `t-termtypeACQMETH` | Acquisition method |
| `t-termtypeCOLL` | Collection name |

---

## 6. Exhibition Records

Exhibitions are named nodes (not blank nodes) with class `am:Exhibition`.

```sparql
?ex a am:Exhibition .
?ex rdfs:label ?label .
?ex am:exhibitionTitle ?title .
?ex am:exhibitionDateStart ?start .
?ex am:exhibitionDateEnd ?end .
?ex am:exhibitionVenue ?venue .
?ex am:exhibitionOrganiser ?personURI .
?ex am:exhibitionLref ?lref .
```

An artwork links to its exhibition history via a blank node on the proxy:
```sparql
?proxy am:exhibition ?exBnode .
?exBnode ... # expand to get dates and link to am:Exhibition
```

---

## 7. Documentation / Bibliography Records

Documentation blank nodes hang off a proxy via `am:documentation`:

```sparql
?proxy am:documentation ?doc .
?doc a am:Documentation .
?doc rdfs:label ?label .
?doc am:documentationTitle ?title .
?doc am:documentationTitleArticle ?article .
?doc am:documentationAuthor ?personURI .
?doc am:documentationPageReference ?pages .
?doc am:documentationSortyear ?year .
?doc am:documentationShelfmark ?shelfmark .
?doc am:documentationTitleLref ?lref .
```

---

## 8. Common Query Patterns

### 8.1 Find an artwork by title keyword

```sparql
SELECT ?proxy ?title ?objectNumber WHERE {
  ?proxy a ore:Proxy .
  ?proxy am:title ?title .
  ?proxy am:objectNumber ?objectNumber .
  FILTER(CONTAINS(LCASE(?title), "amsterdam"))
}
LIMIT 20
```

Prefixes needed: `ore: <http://www.openarchives.org/ore/terms/>`, `am: <http://purl.org/collections/nl/am/>`.

### 8.2 Find an artwork by exact accession number

```sparql
SELECT ?proxy ?title WHERE {
  ?proxy am:objectNumber "A 8" .
  ?proxy am:title ?title .
}
```

### 8.3 Find a person by name

```sparql
SELECT ?person ?name WHERE {
  ?person a am:Person .
  ?person am:name ?name .
  FILTER(CONTAINS(LCASE(?name), "rembrandt"))
}
```

### 8.4 Get all artworks by a specific person (as maker)

```sparql
SELECT ?proxy ?title ?objectNumber WHERE {
  ?proxy am:maker ?makerBnode .
  ?makerBnode rdf:value am:p-31377 .   # Rembrandt Harmensz. van Rijn
  ?proxy am:title ?title .
  ?proxy am:objectNumber ?objectNumber .
}
```

### 8.5 Get full artwork record (proxy + resolved terms)

```sparql
SELECT ?title ?objectNumber ?datStart ?dateEnd ?objectName ?material ?technique WHERE {
  BIND(am:proxy-1000 AS ?proxy)
  ?proxy am:title ?title .
  ?proxy am:objectNumber ?objectNumber .
  OPTIONAL { ?proxy am:productionDateStart ?dateStart . }
  OPTIONAL { ?proxy am:productionDateEnd ?dateEnd . }
  OPTIONAL {
    ?proxy am:objectName ?objNameURI .
    ?objNameURI skos:prefLabel ?objectName .
  }
  OPTIONAL {
    ?proxy am:material ?matURI .
    ?matURI skos:prefLabel ?material .
  }
  OPTIONAL {
    ?proxy am:technique ?techURI .
    ?techURI skos:prefLabel ?technique .
  }
}
```

### 8.6 Get maker name(s) for an artwork

```sparql
SELECT ?makerName ?qualifier WHERE {
  am:proxy-1 am:maker ?makerBnode .
  ?makerBnode rdf:value ?personURI .
  ?personURI am:name ?makerName .
  OPTIONAL { ?makerBnode am:creatorQualifier ?qualifier . }
}
```

### 8.7 Get image URL and catalogue page for an artwork

```sparql
SELECT ?imageURL ?landingPage WHERE {
  am:proxy-1 ore:proxyIn ?aggregation .
  OPTIONAL { ?aggregation edm:object ?imageURL . }
  OPTIONAL { ?aggregation edm:landingPage ?landingPage . }
}
```

### 8.8 Find artworks by subject/material/technique

```sparql
# By subject keyword
SELECT ?proxy ?title WHERE {
  ?proxy am:contentSubject ?subj .
  ?subj skos:prefLabel ?subjLabel .
  FILTER(CONTAINS(LCASE(?subjLabel), "portret"))
  ?proxy am:title ?title .
} LIMIT 20
```

### 8.9 Get full person biography

```sparql
SELECT ?name ?biography ?birthDate ?deathDate ?birthPlace ?deathPlace WHERE {
  BIND(am:p-31377 AS ?person)
  ?person am:name ?name .
  OPTIONAL { ?person am:biography ?biography . }
  OPTIONAL { ?person am:birthDateStart ?birthDate . }
  OPTIONAL { ?person am:deathDateStart ?deathDate . }
  OPTIONAL { ?person am:birthPlace ?birthPlace . }
  OPTIONAL { ?person am:deathPlace ?deathPlace . }
}
```

### 8.10 Browse artworks from a specific production period

```sparql
SELECT ?proxy ?title ?dateStart ?dateEnd WHERE {
  ?proxy a ore:Proxy .
  ?proxy am:title ?title .
  ?proxy am:productionDateStart ?dateStart .
  ?proxy am:productionDateEnd ?dateEnd .
  FILTER(xsd:integer(?dateStart) >= 1600 && xsd:integer(?dateEnd) <= 1700)
} LIMIT 20
```

---

## 9. Namespace Prefixes (use in all SPARQL queries)

```sparql
PREFIX am:    <http://purl.org/collections/nl/am/>
PREFIX rdf:   <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs:  <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos:  <http://www.w3.org/2004/02/skos/core#>
PREFIX ore:   <http://www.openarchives.org/ore/terms/>
PREFIX edm:   <http://www.europeana.eu/schemas/edm/>
PREFIX xsd:   <http://www.w3.org/2001/XMLSchema#>
```

---

## 10. Scale & Coverage

| Class | Count |
|-------|-------|
| `am:Dimension` (measurements) | 154,828 |
| `edm:WebResource` (media) | 146,895 |
| `am:Locat` (location records) | 137,895 |
| `am:Reproduction` (image records) | 114,463 |
| `am:Maker` (maker links) | 80,635 |
| `edm:PhysicalThing` = artworks | **73,447** |
| `ore:Proxy` = artwork records | **73,447** |
| `am:Person` (people/artists/orgs) | 66,966 |
| `am:Documentation` (bibliography) | 38,613 |
| `skos:Concept` (thesaurus terms) | 28,047 |
| `am:Alternativenumber` | 21,521 |
| `am:Exhibition` | 10,745 |

The collection is Amsterdam-focused: Dutch Golden Age prints, paintings, silverware, applied arts, and city history from the Middle Ages to the 20th century. Titles are primarily in Dutch (`@nl`), some in English (`@en`).

---

## 11. Tips & Pitfalls

1. **Blank nodes cannot be addressed by URI.** Dimensions, locations, reproductions, makers, and exhibitions off a proxy are blank nodes — always traverse from the proxy with a `?proxy am:dimension ?dim . ?dim ...` pattern.

2. **Multiple titles per object.** A proxy can have several `am:title` values (Dutch, English, or alternative titles). Use `GROUP_CONCAT` or `OPTIONAL` and expect multiple rows.

3. **Term labels are in Dutch.** `skos:prefLabel` values are tagged `@nl`. Use `FILTER(LANG(?label) = "nl")` if you get duplicates.

4. **`am:Person` includes organizations.** Institutions, foundations, and committees are also stored as `am:Person` nodes (e.g. "Museum het Rembrandthuis", "Vereeniging Rembrandt").

5. **`entity_search` is fuzzy but limited.** It searches label-like predicates but returns a small number of results and scores are not very discriminating. For precise lookup always use `sparql_query` with `FILTER(CONTAINS(...))`.

6. **Date fields are strings, not xsd:date.** Use `xsd:integer(?dateStart)` casting for numeric year comparisons.

7. **`am:relatedObjectReference` is very dense** (213K triples). Avoid unfiltered traversals; always anchor on a specific proxy.

8. **The `am:selected` flag on Person** (`"True"`/`"False"`) does not reliably identify major artists — most entries are `"False"` even for well-known figures. Ignore it for ranking purposes.

9. **`priref` numbers are not sequential by importance.** Use accession `objectNumber` (e.g. `"A 8"`, `"SA 7360.1"`) when users refer to specific catalogue entries.

10. **Image URLs** from `edm:object` on aggregation nodes point to the museum's own image server (`am.adlibhosting.com`). These may or may not be publicly accessible.
