from dataclasses import dataclass

from lightworld.tools.entity_prompt_extractor import EntityPromptExtractor
from lightworld.simulation.oasis_profile_generator import OasisProfileGenerator
from lightworld.simulation.simulation_config_generator import SimulationConfigGenerator
from lightworld.simulation.simulation_population import SimulationPopulationBuilder
from lightworld.graph.social_relation_graph import SocialRelationGraphCompiler
from lightworld.graph.zep_entity_reader import ZepEntityReader


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
