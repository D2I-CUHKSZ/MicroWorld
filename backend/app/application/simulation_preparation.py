from dataclasses import dataclass

from ..utils.entity_prompt_extractor import EntityPromptExtractor
from ..utils.oasis_profile_generator import OasisProfileGenerator
from ..utils.simulation_config_generator import SimulationConfigGenerator
from ..utils.simulation_population import SimulationPopulationBuilder
from ..utils.social_relation_graph import SocialRelationGraphCompiler
from ..utils.zep_entity_reader import ZepEntityReader


@dataclass
class SimulationPreparationServices:
    reader: ZepEntityReader
    population_builder: SimulationPopulationBuilder
    prompt_extractor: EntityPromptExtractor
    profile_generator: OasisProfileGenerator
    config_generator: SimulationConfigGenerator
    relation_graph_compiler: SocialRelationGraphCompiler


class SimulationPreparationFactory:
    def create(self, graph_id: str) -> SimulationPreparationServices:
        return SimulationPreparationServices(
            reader=ZepEntityReader(),
            population_builder=SimulationPopulationBuilder(),
            prompt_extractor=EntityPromptExtractor(),
            profile_generator=OasisProfileGenerator(graph_id=graph_id),
            config_generator=SimulationConfigGenerator(),
            relation_graph_compiler=SocialRelationGraphCompiler(),
        )
