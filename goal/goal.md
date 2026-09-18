# Darukaa.Earth: AI Biodiversity Intelligence Chatbot Challenge

Welcome to the Darukaa.Earth Hackathon. This challenge is designed to evaluate your
ability to build intelligent, knowledge-driven AI systems that can reason about real-world
environmental problems.

We are not looking for UI-heavy applications. We are looking for depth of thinking, system
design, and scientific reasoning.

## Objective

Build an AI-powered conversational system that:

● Maintains a structured knowledge base of biodiversity and environmental metrics  
● Understands user queries about ecosystems, land, and climate conditions  
● Generates actionable, non-obvious recommendations to improve biodiversity  
● Supports every recommendation with scientific reasoning and evidence

## Core Requirements

### 1. Knowledge System (Critical)

Your system must include a retrievable knowledge layer, not just prompts.

It should cover:

● Soil health (pH, organic carbon, moisture)  
● Land use / land cover  
● Biodiversity indicators (species richness, habitat diversity)  
● Climate factors (temperature, rainfall)  
● Human impact (pollution, deforestation)

**Expectation:**

● Use RAG, embeddings, vector databases, or structured datasets  
● Index research papers, reports, or environmental datasets  
● Clearly show how knowledge is retrieved and used

### 2. Conversational Intelligence

Your system should:

● Ask clarifying questions when inputs are incomplete  
● Handle multi-turn conversations with memory  
● Adapt responses based on context

**Example:**

User: “Biodiversity is declining on my land”

System: “Can you provide soil organic carbon %, rainfall pattern, and land use
type?”

### 3. Evidence-Backed Recommendations (Mandatory)

Each recommendation must include:

● What to do  
● Why it works (scientific reasoning)  
● Which environmental metric improves  
● Reference to a study/report/model

**Expected Quality:**

“Introduce legume-based cover crops → increases soil organic carbon by
~15–25% over 2–3 years (FAO studies), improving microbial diversity and
pollinator support.”

**Not acceptable:**

“Use sustainable practices”

### 4. Multi-Metric Reasoning

Your system must connect multiple variables:

● Soil health ↔ biodiversity  
● Water availability ↔ species survival  
● Land use ↔ habitat fragmentation

This is the core differentiator—no single-variable answers.

### 5. Input Handling

Support at least:

● Text input (mandatory)  
● Structured input (JSON or similar)

Bonus:

● Geo-coordinates or spatial context

### 6. Output Quality

Each response must clearly include:

● Recommendation  
● Impacted metrics  
● Time horizon (short / medium / long term)  
● Confidence level (optional but valuable)

**Example Use Case**

Input:

● Soil organic carbon: 0.3%  
● Rainfall: low  
● Crop: monoculture wheat  
● Region: semi-arid

Expected Output:

● Suggest agroforestry / intercropping  
● Explain impact on soil carbon and biodiversity  
● Provide measurable improvement estimates  
● Reference credible sources such as  
  ○ Food and Agriculture Organization  
  ○ Intergovernmental Panel on Climate Change

## Evaluation Criteria

### 1. Depth of Reasoning (30%)

● Are recommendations non-obvious?  
● Do they combine multiple environmental variables?

### 2. Scientific Grounding (25%)

● Are claims backed by credible sources?  
● Is reasoning accurate and explainable?

### 3. Knowledge System Design (20%)

● Use of RAG / vector DB / structured datasets  
● Clarity of knowledge retrieval pipeline

### 4. Conversational Intelligence (15%)

● Context awareness  
● Follow-up questioning  
● Memory handling

### 5. Output Clarity (10%)

● Structured, readable, and actionable responses

## Constraints

● No generic LLM-only solutions  
● No shallow or obvious recommendations

## Constraints

• Must demonstrate knowledge grounding + reasoning.  
• Must handle at least 3 environmental variables together.

## Goal

Build a system that behaves like an AI environmental scientist, not a chatbot.

## Documents Submission Guidelines

Submit one Word document (.docx) through the job where you applied, using the My Jobs page / Applied Job page and its document-submission option.

The Word document must include:

1 GitHub repository link for the completed project.  
2 Live demo URL, where applicable.  
3 A brief README.md overview covering the architecture, database/schema, local setup, and CI/CD details.  
4 Any other links, credentials, or notes required to review and run the submission.

## Repository access instructions

If the GitHub repository is private, grant access to the following accounts and include the repository link in the Word document:

• ankita.dasgupta@darukaa.com  
• harsh.kumar@darukaa.com  
• utkarsh.gauniyal@darukaa.com  
• guneet.mutreja@darukaa.com

If the repository is public, provide the repository link only; access invitations are not required.

All submission details and links must be included in the Word document submitted through the applied-job page.