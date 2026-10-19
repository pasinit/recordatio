"""Prompt templates for knowledge extraction."""

EXTRACTION_TARGET_RULES = """The input is split into a [context] block and a [target] block.
Extract only from the [target] block. Use [context] solely to resolve
references such as pronouns, shorthand, and prior topics. Facts already stated
in [context] are not new; do not re-emit them, even paraphrased.
"""

ATOMIC_MEMORY_PROMPT_TEMPLATE = """You extract reusable atomic memories from agent conversations.

{target_rules}
Return only durable facts, specific events, or genuinely reusable methods from
the target. Do not extract entity mentions, attributes, relations, ontology
schema, topics, or schema proposals in this pass.

Each "units" item is:
{{"id": "<optional local id>", "type": "semantic|episodic|procedural",
 "content": "<concise self-contained statement>", "provenance": "extraction",
 "confidence": 0.0-1.0,
 "evidence": [{{"source": "target", "quote": "..."}}]}}.

DO NOT extract units that describe conversation mechanics, including resumed
sessions, memory state, acknowledgements, greetings, or prior context gaps.

DO NOT turn operational/status reports into procedural memories. Reports can
contain useful new facts and events; extract those concrete facts/events, but
omit report scaffolding and restated workflows such as recap formatting,
source-checking steps, delivery instructions, or task-execution bookkeeping
unless the target contains a genuinely new reusable method not already implied
by context.

PER-ENTITY STATE IN A DELIVERABLE IS DURABLE — extract it. A tracker, pipeline,
recap, or status brief carries the current STATE of the entities it tracks, and
that state IS the point of the deliverable. Even though it lives inside a report,
you MUST extract each per-entity state record as a durable unit (prefer type
"episodic") so continuity survives from one run to the next. Emit one
self-contained unit per tracked entity/row, carrying the entity identifier, its
key attributes, and its current status. This carve-out is distinct from the
formatting/workflow scaffolding above, which you still omit. Examples:
- "User submitted a job application to Acme Foundation (role: Analyst,
  date: 2026-08-14); current status: under review."
- "Property at 3 Demo Street (portal id 100001, £2,000 pcm, 55 sqm, floor
  2); pipeline status: still tracking as of 2026-08-16."

THE USER ENTITY:
When a memory concerns the person sending [user] messages, refer to that person
as "{user_entity_label}".

Respond with ONLY a JSON object. Required top-level keys:
units. Use [] for empty sections.
"""

ONTOLOGY_FACTS_PROMPT_TEMPLATE = """You extract ontology facts from agent conversations.

{target_rules}
Return entity mentions, scalar attributes, and relations only. Do not extract
atomic memory units, topics, or schema proposals in this pass. Prefer the
existing ontology below. If no existing type fits, leave the type blank or use
a conservative reusable name only when the target gives strong evidence. Do
not overfit domains by inventing one-off labels.

THE USER ENTITY:
The person sending [user] messages is represented by "{user_entity_label}".
When the target reveals personal facts about this person, use
"{user_entity_label}" as the subject and include it as an entity mention.

KNOWN ENTITY TYPES:
{entity_types_block}

KNOWN ATTRIBUTE TYPES:
{attribute_types_block}

KNOWN RELATION TYPES:
{relation_types_block}

"entity_mentions" items are:
{{"id": "<optional local id>", "label": "<canonical name>",
 "type": "<known entity type if one fits>", "subtype": "<optional narrower type>",
 "aliases": ["..."], "identifiers": {{"namespace": "value"}},
 "canonical_key": "<optional stable key>", "confidence": 0.0-1.0,
 "evidence": [{{"source": "target", "quote": "..."}}]}}.

"attributes" items are:
{{"subject": "<entity label>", "subject_id": "<optional entity mention id>",
 "attribute": "<attribute type>", "value": "<short scalar value>",
 "provenance": "extraction", "confidence": 0.0-1.0,
 "evidence": [{{"source": "target", "quote": "..."}}]}}.

"relations" items are:
{{"subject": "<entity label or short value>",
 "subject_id": "<optional entity mention id>", "predicate": "<relation type>",
 "object": "<entity label or short value>",
 "object_id": "<optional entity mention id>", "provenance": "extraction",
 "confidence": 0.0-1.0,
 "evidence": [{{"source": "target", "quote": "..."}}]}}.

Relation endpoints MUST be short atomic labels or scalar values, such as
"User", "Sunnydale", "Acme", or "2000 pcm". Never use sentence fragments, whole
knowledge units, listing descriptions, report sections, instructions, or
context dumps as relation subjects or objects. If a relationship cannot be
expressed with short endpoints, omit the relation.

Avoid arbitrary ontology drift. Do not invent operational or workflow
predicates such as "searches_on", "daily_recap_for", "validated_against",
"listing_agent", or similar bookkeeping labels.

Respond with ONLY a JSON object. Required top-level keys:
entity_mentions, attributes, relations. Use [] for empty sections.
"""

SCHEMA_PROPOSAL_PROMPT_TEMPLATE = """You review the target for reusable ontology schema proposals.

{target_rules}
Return only schema proposals and topic labels. Do not return accepted facts,
entity mentions, attributes, relations, or atomic memory units in this pass.
Propose a new schema item only when the target gives strong evidence for a
durable, reusable category that is not already covered by the known schema.
Prefer existing schema and keep domain-specific overfitting out of proposals.

KNOWN ENTITY TYPES:
{entity_types_block}

KNOWN ATTRIBUTE TYPES:
{attribute_types_block}

KNOWN RELATION TYPES:
{relation_types_block}

KNOWN TOPICS:
{topics_block}

"schema_proposals" items are:
{{"kind": "entity_type|attribute_type|relation_type|topic",
 "name": "<short reusable name>", "description": "<why it is reusable>",
 "applies_to": "<optional entity type>", "provenance": "extraction",
 "confidence": 0.0-1.0,
 "evidence": [{{"source": "target", "quote": "..."}}]}}.

Relation and attribute type names must be lower_snake_case predicates. Topics
are labels that classify the target. Put existing topic labels in "topics".
Propose new topic labels as schema_proposals with kind "topic".

Respond with ONLY a JSON object. Required top-level keys:
schema_proposals, topics. Use [] for empty sections.
"""

# Backward-compatible import name for older callers that imported the original
# single prompt constant directly.
SYSTEM_PROMPT_TEMPLATE = ATOMIC_MEMORY_PROMPT_TEMPLATE.replace("{target_rules}", EXTRACTION_TARGET_RULES)
