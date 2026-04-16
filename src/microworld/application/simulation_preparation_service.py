from dataclasses import dataclass

from microworld.tools.entity_prompt_extractor import EntityPromptExtractor
from microworld.simulation.oasis_profile_generator import OasisProfileGenerator
from microworld.simulation.simulation_config_generator import SimulationConfigGenerator
from microworld.simulation.simulation_population import SimulationPopulationBuilder
from microworld.graph.social_relation_graph import SocialRelationGraphCompiler
from microworld.graph.zep_entity_reader import ZepEntityReader


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
