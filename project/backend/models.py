from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field


class SoilParameters(BaseModel):
    organic_carbon: Optional[Union[float, str]] = Field(None, description="Soil organic carbon percentage (e.g. 0.3 or '0.3%')")
    moisture: Optional[str] = Field(None, description="Moisture level (e.g. 'low', 'adequate', 'high')")
    ph: Optional[Union[float, str]] = Field(None, description="Soil pH level (e.g. 6.5, 'acidic', 'alkaline')")
    texture: Optional[str] = Field(None, description="Soil texture (e.g. 'sandy loam', 'clay')")
    erosion_level: Optional[str] = Field(None, description="Observed erosion level")


class ClimateParameters(BaseModel):
    rainfall: Optional[str] = Field(None, description="Rainfall pattern (e.g. 'low', '350mm/year', 'erratic')")
    temperature: Optional[str] = Field(None, description="Temperature regime (e.g. 'high', 'moderate', 'semi-arid')")
    drought_frequency: Optional[str] = Field(None, description="Frequency of drought events")


class LandUseParameters(BaseModel):
    crop: Optional[str] = Field(None, description="Current crop(s) (e.g. 'wheat', 'maize', 'pasture')")
    system: Optional[str] = Field(None, description="Farming system (e.g. 'monoculture', 'rotational', 'agroforestry')")
    tillage: Optional[str] = Field(None, description="Tillage practices (e.g. 'conventional', 'minimum', 'zero')")
    land_cover: Optional[str] = Field(None, description="General land cover type")


class BiodiversityParameters(BaseModel):
    habitat_diversity: Optional[str] = Field(None, description="Habitat diversity rating (e.g. 'low', 'moderate')")
    species_richness: Optional[str] = Field(None, description="Species richness status")
    pollinator_status: Optional[str] = Field(None, description="Pollinator presence or abundance")
    vegetation_cover: Optional[str] = Field(None, description="Percentage or density of natural vegetation")


class EnvironmentalState(BaseModel):
    region: Optional[str] = Field(None, description="Geographical or ecological region")
    soil: SoilParameters = Field(default_factory=SoilParameters)
    climate: ClimateParameters = Field(default_factory=ClimateParameters)
    land_use: LandUseParameters = Field(default_factory=LandUseParameters)
    biodiversity: BiodiversityParameters = Field(default_factory=BiodiversityParameters)
    human_impact: Optional[str] = Field(None, description="Noted anthropogenic pressures (e.g. pesticide overuse, grazing)")
    raw_text: Optional[str] = Field(None, description="Raw user prompt if input was unstructured text")


class InvestigativeHypothesis(BaseModel):
    hypothesis: str = Field(..., description="Scientific hypothesis to investigate, formulated without declaring unproven facts")
    query: str = Field(..., description="Targeted retrieval search query optimized for scientific documents")
    target_metrics: List[str] = Field(default_factory=list, description="Environmental metrics this query targets")
    rationale: str = Field(..., description="Why this evidence is needed before generating a recommendation")


class PlannerOutput(BaseModel):
    environmental_assessment: str = Field(..., description="Concise assessment of the given ecosystem state")
    key_constraints: List[str] = Field(..., description="Identified environmental constraints to cross-examine")
    hypotheses: List[InvestigativeHypothesis] = Field(..., description="3 to 5 targeted hypothesis queries")


class EvidenceChunk(BaseModel):
    evidence_id: str = Field(..., description="Unique evidence ID (e.g. E001, E002)")
    doc_title: str = Field(..., description="Document title / source")
    section: Optional[str] = Field(None, description="Section or chapter heading")
    page: Optional[Union[int, str]] = Field(None, description="Page or chunk index")
    score: float = Field(..., description="Vector similarity score")
    text: str = Field(..., description="Extracted chunk text")
    matched_query: Optional[str] = Field(None, description="Query that retrieved this chunk")


class EvidenceGroundedClaim(BaseModel):
    claim: str = Field(..., description="Scientific statement or deduction")
    evidence_ids: List[str] = Field(default_factory=list, description="IDs of evidence chunks supporting this claim")


class SpecificRecommendation(BaseModel):
    what_to_do: str = Field(..., description="Actionable intervention (e.g. Introduce legume-based cover crops)")
    why_it_works: str = Field(..., description="Scientific reasoning explaining mechanisms across variables (e.g. increases soil microbial biomass and nitrogen fixation)")
    impacted_metrics: List[str] = Field(default_factory=list, description="Specific environmental metrics improved")
    measurable_outcome: str = Field(..., description="Quantitative or qualitative metric impact (e.g. increases soil organic carbon by ~15-25% over 2-3 years)")
    reference_citation: str = Field(..., description="Reference to credible study/report/model (e.g. FAO 2019, IPCC 2019, IPBES)")
    time_horizon: str = Field(..., description="Short-term (1-2 yrs) / Medium-term (3-5 yrs) / Long-term (5-10+ yrs)")
    confidence_level: str = Field(..., description="High / Medium / Moderate")
    evidence_ids: List[str] = Field(default_factory=list, description="Retrieved evidence IDs supporting this recommendation")



class ValidationVerdict(str):
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED_NUMBER = "UNSUPPORTED_NUMBER"
    CONTEXT_MISMATCH = "CONTEXT_MISMATCH"
    WEAKENED = "WEAKENED"


class ValidationFlag(BaseModel):
    original_claim: str = Field(..., description="The original claim text before validation")
    verdict: str = Field(..., description="SUPPORTED | UNSUPPORTED_NUMBER | CONTEXT_MISMATCH | WEAKENED")
    corrected_claim: Optional[str] = Field(None, description="Softened/corrected claim if not fully supported")
    reason: str = Field(..., description="Why this verdict was assigned")
    evidence_ids_checked: List[str] = Field(default_factory=list, description="Evidence IDs that were checked for this claim")


class RecommendationOutput(BaseModel):
    title: str = Field(..., description="Headline of the recommended intervention")
    primary_intervention: str = Field(..., description="Clear, non-obvious, actionable intervention")
    environmental_assessment: str = Field(..., description="Executive summary of interacting ecosystem constraints")
    scientific_reasoning: List[EvidenceGroundedClaim] = Field(..., description="Step-by-step grounded rationale linking metrics")
    recommendations: List[SpecificRecommendation] = Field(default_factory=list, description="Structured list of evidence-backed recommendations")
    action_plan: List[str] = Field(..., description="Specific practical implementation steps")
    impacted_metrics: Dict[str, str] = Field(..., description="Specific metric impacts with expected magnitudes/timelines")
    time_horizon: str = Field(..., description="Implementation and impact timeline (e.g., Short-term 1-2 yrs, Medium 3-5 yrs)")
    confidence: str = Field(..., description="Scientific confidence rating (High / Medium / Moderate)")
    trade_offs: List[EvidenceGroundedClaim] = Field(..., description="Intervention trade-offs and mitigation strategies")
    retrieved_evidence: List[EvidenceChunk] = Field(default_factory=list, description="Auditable evidence items with citations")
    validation_flags: List[ValidationFlag] = Field(default_factory=list, description="Per-claim audit results from LLM #3 validator")
    validation_applied: bool = Field(False, description="Whether claim-evidence validation was run on this output")
    data_availability_notice: Optional[str] = Field(None, description="Explicit geographic provenance and evidence availability notice (e.g. if local trial data for Bengaluru is unavailable in the database)")




class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"] = Field(..., description="Who sent this message")
    content: str = Field(..., description="Message text")


class InputValidationResult(BaseModel):
    is_sufficient: bool = Field(..., description="True if input has sufficient context for grounded multi-variable reasoning, False if too vague")
    clarifying_question: Optional[str] = Field(None, description="Clarifying question asking user for missing key parameters")
    missing_variables: List[str] = Field(default_factory=list, description="List of missing parameters (e.g. ['soil organic carbon %', 'rainfall pattern', 'land use type'])")
    detected_context: Dict[str, Any] = Field(default_factory=dict, description="Variables detected from user input")
    rewritten_query: Optional[str] = Field(None, description="Full merged query reconstructed from conversation history — used for downstream planner")


class PipelineTiming(BaseModel):
    retrieval_ms: int = Field(0, description="Time spent querying Qdrant and re-ranking in ms")
    retrieval_time_s: float = Field(0.0, description="Time spent in retrieval in seconds")
    thinking_ms: int = Field(0, description="Time spent in LLM planning and reasoning in ms")
    thinking_time_s: float = Field(0.0, description="Time spent in LLM reasoning in seconds")
    total_ms: int = Field(0, description="Total pipeline latency in ms")
    total_time_s: float = Field(0.0, description="Total pipeline latency in seconds")
    rewriting_ms: int = Field(0, description="Time spent in query rewriting/validation in ms")


class PipelineResponse(BaseModel):
    needs_clarification: bool = Field(False, description="Whether clarifying questions are needed before planning")
    clarification: Optional[InputValidationResult] = Field(None, description="Clarification details if input was insufficient")
    planner: Optional[PlannerOutput] = Field(None, description="Planner output with hypotheses and queries")
    recommendation: Optional[RecommendationOutput] = Field(None, description="Final evidence-grounded recommendation")
    timing: Optional[PipelineTiming] = Field(None, description="Execution timing breakdown")


class UserInputRequest(BaseModel):
    text: Optional[str] = Field(None, description="Unstructured natural language input")
    structured: Optional[Dict[str, Any]] = Field(None, description="Structured environmental variables")
    conversation_history: List[ConversationMessage] = Field(
        default_factory=list,
        description="Prior turns in this chat session — used for context-aware query rewriting"
    )

