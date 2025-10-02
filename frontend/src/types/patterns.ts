export type PatternMeta = {
  id_pattern: string;
  nom_fr: string;
  categorie: "Raisonnement" | "Action" | "Amélioration" | "Orchestration";
  fonctionnement_court: string;
  utilite_concrets: string[];
  detail_technique: string;
  implication_backend: string;
  compromis_cout_latence: string;
  has_demo: boolean;
  tags: string[];
};
