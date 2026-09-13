# Veille & Rapport de Cadrage

## Page de titre

**Nom et prénom :** Laces Vitomir<br>
**Numéro de groupe :**  9<br>
**Nom de projet :** Alloquence<br>
**Liste des membres de votre équipe :** Huang Victor, Mabrouki Rayane, Mahious Jugurta, Aidibe Hassan<br>

## La problématique

Malgré l'essor du numérique, 42 % des réservations se font toujours par téléphone (UMIH, 2026). Ce canal historique, délaissé par la digitalisation, génère aujourd'hui une double perte pour les commerçants : du chiffre d'affaires envolé vers la concurrence à chaque appel non décroché, et des marges gâchées par les réservations non honorées.

Nous proposons la solution suivante : une intelligence artificielle vocale pour prendre les appels à leur place, on s'assure de ne plus rater aucun client.

## Analyse du marché

Le marché de la prise de réservation en restauration fait face à un paradoxe majeur : bien que la dématérialisation progresse, le téléphone reste le canal historique privilégié, tout en générant une perte financière considérable durant les heures de rush.

**Les faits chiffrés du marché :**

* **42 % des réservations** s'effectuent encore directement par téléphone (Source : UMIH, 2026).

* **8 à 12 % du chiffre d'affaires annuel** est perdu en raison des appels non décrochés pendant les services (Source : Fédération Française de Restauration).

* **80 % des clients** ne rappellent pas après être tombés sur un appel manqué ou un répondeur, et se tournent vers un établissement concurrent (Source : Obs. Services Clients 2023).

**La concurrence**

1. Le “fait à la main”:

   * **Forces :** Gratuit, contact humain direct

    * **Faiblesses :** Indisponibilité pendant le rush, risque d’erreur de notation, perte de clients le soir et le week-end, consomme le temps et les ressources d'un salarié.

1. **Répondia**

    * **Forces :** Automatise la prise de réservation et relance automatiquement le client par message texte après un appel manqué.

    * **Faiblesses :** Canal unique (WhatsApp) : ne traite pas directement l'appel vocal en direct. Le client doit obligatoirement avoir l'application WhatsApp et accepter d'échanger par écrit.

1. **Ryyng**

    * **Forces :** Décroche les appels téléphoniques et fait de la prise de réservation vocale.

    * **Faiblesses :** Latence élevée lors des échanges (effet robotique/pauses inconfortables) et ne se déclenche qu'après coup sur les appels manqués (pas en direct sur 100 % des flux).

1. **Yumcall**

    * **Forces :** Prise de rendez-vous/réservation par agent vocal IA.

    * **Faiblesses :** Volume restreint & Souveraineté : Offre limitée à 150 minutes/mois et solution opérée depuis la Lettonie (interrogations sur la gestion des données/RGPD).


**Méthodologie de veille**

1. **Objectifs de veille**

    * **Technologique :** Suivre les sorties de nouveaux modèles Speech-to-Text (STT) et Text-to-Speech (TTS) pour réduire la latence.

    * **Concurrentielle :** Surveiller les tarifs et les fonctionnalités des nouveaux assistants vocaux sur le marché.

    * **Usages :** Analyser l’acceptabilité de l’intelligence artificielle par les clients finaux

2. **Ressources et outils**

    * **Moteurs de recherche :** Utilisation de Google avec des termes spécifiques (exemple : intelligence artificielle vocale, commerces de proximité)

    * **Analyse de la presse technologique :** Consultation régulière de sites comme Product Hunt pour repérer les nouveaux outils STT/TTS et les agents vocaux émergents.

    * **Veille communautaire (réseaux sociaux) :** Reddit, X (Twitter), forums, Agent IA, Youtube, Newsletter (exemple : TLDR), Flux RSS, Podcasts

3. **Auto-critique**

    * **Points forts :** Accès gratuit à une information mondiale et actualisée en temps réel

    * **Limites :** L’abondance d’informations sur internet rend parfois difficile le tri entre les vraies innovations et le marketing. Pour compenser on a testé les outils directement.


## Analyse de la problématique

**Identification et segmentation des parties prenantes**

* **Le commerçant :** Propriétaire du restaurant ou du cabinet de kiné. Son but : rentabilité et sérénité.
* **Le staff :** Serveurs, secrétaires. Leur but : ne plus être interrompus.
* **Le client final :** Nous par exemple. Notre but : une réponse immédiate, 24h/24.

**Où se situe la principale source de douleur ?**

* **Douleur du restaurateur / staff :** Changer de tâche (passer de la salle au téléphone) demande un effort mental qui fatigue et provoque des erreurs (oublis de commande, erreur de réservation)

* **Douleur du client :** S’il appelle trois fois sans réponse, il a l’impression que l’établissement est mal géré ou fermé.

**D’où vient l’argent ?**

* **Le No-Show :** Notre solution peut envoyer un SMS de rappel automatique. Un rendez-vous non honoré chez un kiné, c’est 30€ à 50€ de perte.

* **Le coût d’opportunité :** Un restaurateur qui rate 5 appels un samedi soir perd potentiellement une table de 4 personnes. À 35€ par tête, c’est 140€ de chiffre d’affaires évaporé en 2 minutes.

* **Le coût du personnel :** Engager un standardiste coûte environ 2500€/mois (charges comprises). Notre solution coûte une fraction de ce prix.

**Résultat de l’enquête**

L'entretien avec un restaurateur, a permis de valider notre problématique tout en soulevant des exigences cruciales pour l'acceptabilité de la solution. 

1. **Citations clés et Pain Points**

   * **Sur le manque de ressources :** "Il n'a pas les moyens d'engager un hôte/hôtesse" pour gérer uniquement les appels.

   * **Sur la perte d'efficacité :** "Pendant les rushs, il n'a pas le temps de prendre ses appels". 

   * **Sur l'exigence de qualité (Le "zéro latence") :** "Prendre un appel ne doit pas prendre 4 ans", le restaurateur exige un minimum de décalage pour ne pas frustrer ses clients, notamment les plus âgés.

2. **Ce que le restaurateur attend de l’intelligence artificielle**

    * **Naturalité et personnalisation :** Il refuse l'effet "script robotique". L'intelligence artificielle doit s'exprimer "comme si c'était lui qui parlait" et être capable de donner des informations spécifiques. 

    * **Hybridation Homme/intelligence artificielle :** Le restaurateur ne veut pas que l'intelligence artificielle remplace tout. Il souhaite qu'elle prenne le relais uniquement quand il est occupé, afin de pouvoir continuer à "discuter avec ses clients" lorsqu'il est disponible.
 
    * **Zéro maintenance :** La solution doit fonctionner de manière totalement autonome sans qu'il ait à "intervenir sur la solution" au quotidien. 

3. **Besoins matériels et intégration**

    * L'outil doit être centralisé sur le PC de caisse actuel pour rester statique et accessible à tous les employés. 

    * L'interface doit être responsive sur téléphone secondaire et permettre des accès limités pour que chaque employé puisse gérer les tables en temps réel.

Cette interview confirme que le besoin n'est pas seulement de "répondre au téléphone", mais de le faire avec une **finesse humaine** et une **vitesse technique** irréprochable pour préserver l'image de marque du commerce. 

**Hiérarchisation des priorités**

1. **Performance et Réactivité (Latence) :** L'intelligence artificielle doit répondre instantanément pour éviter la frustration des clients, surtout les plus âgés qui pourraient raccrocher face à un silence.

1. **Fiabilité de la compréhension :** Indispensable pour gérer automatiquement les réservations sans erreur sur les noms ou les dates.

1. **Expérience utilisateur client final :** L’intelligence artificielle ne doit pas avoir un effet robot.

1. **Synchronisation et Disponibilité :** Assurer que l'agenda sur le PC de caisse soit mis à jour en temps réel pour que les employés voient les réservations immédiatement.

## Proposition de solution

**Les différents types d’utilisateurs de notre plateforme**

* **Les gérants des commerces de proximités** (ex : restaurateur, kiné, ostéopathe, garagiste), nous tenons à préciser que notre POC va se concentrer sur les restaurateurs dans un premier temps

* **Le staff des commerces de proximités** (ex: serveur, secrétaire)

* **Les clients des commerces de proximités**
    * **Les clients pressés :** Qui veulent réserver en 30 secondes sans attendre. 

    * **Les clients âgés :** Qui sont attachés au téléphone mais qui ne doivent pas être déstabilisés par un "effet robot" ou des temps de réponse trop longs.

**“use cases”**

* Le gérant du restaurant est occupé à accueillir les clients dans son établissement, le téléphone sonne sans arrêt il ne peut pas y répondre, notre solution gère ses appels à sa place.

* Le serveur est en train de faire son service, toute l’équipe est occupée, le gérant lui demande de répondre au téléphone, il ne peut pas parce qu’il a trop de gens à servir. Notre solution gère les appels à sa place, et il peut continuer de livrer un service de qualité.

* Un client veut réserver une table pour 12:30 ça fait déjà la quatrième fois qu’il tombe sur le répondeur du restaurant, il est frustré. Notre solution aurait répondu dès son premier appel et la table aurait été réservée en fonction des disponibilités.

**Maquettes / wireframes**

* Staff ![alt text](image.png)

* Gérant ![alt text](image-1.png)

* Client final ![alt text](image-4.png)

**Architecture**

![alt text](image-5.png)

**Fonctionnalitées**

1. **Agent vocal intelligent et adaptatif**

   * **Réponse instantanée et personnalisée :** l'IA décroche immédiatement avec un message de bienvenue propre à l'établissement.

   * **Traitement du langage naturel :** l'IA comprend les demandes variées (réservations, commandes ou simples questions).

   * **Gestion des connaissances locales :** l'IA répond aux questions spécifiques comme « Est-ce halal ? » ou « Quel est le menu du jour ? » en consultant une base de données mise à jour par le gérant.

   * **Support multilingue :** l'IA détecte la langue de l'appelant et peut lui répondre (français et anglais au minimum), pour ne pas perdre un client non francophone. (V2)

   * **Gestion des incompréhensions (fallback) :** si l'IA ne comprend pas après deux tentatives, elle ne reste jamais en boucle ; elle propose une issue (reformulation, transfert) et, si rien n'aboutit, bascule vers le filet de sécurité.

   * **Filet de sécurité — demande de rappel garantie :** règle d'or, un appel ne se termine jamais en silence ou en sonnerie infinie. Si l'IA et le staff n'ont pas pu traiter la demande, l'IA capture le motif et le numéro de l'appelant, crée une demande de rappel prioritaire dans le dashboard, et notifie le restaurateur et le client par SMS.

1. **Système de réservation automatisé**

   * **Vérification de disponibilité en temps réel :** le système interroge instantanément l'agenda pour confirmer si une table est libre.

   * **Résolution de conflits :** en cas d'indisponibilité, l'IA propose proactivement des alternatives au client.

   * **Confirmation multicanale :** une fois la réservation validée, un SMS de confirmation est automatiquement envoyé au client pour sécuriser la venue.

1. **Système de commande automatisé**

   * **Prise de commande et récapitulatif omnicanal :** l'IA prend la commande du client par téléphone et lui envoie instantanément un SMS détaillant son panier.

   * **Sécurisation des paiements :** le SMS intègre un lien de paiement sécurisé permettant au client de régler sa commande à distance, protégeant le commerçant.

   * **Validation conditionnée au paiement :** la préparation en cuisine n'est déclenchée qu'une fois le paiement confirmé.

   * **Réduction de la friction de paiement :** prise en charge d'Apple Pay, Google Pay et Stripe Link.

   * **Passerelle de secours numérique :** en cas de doute, le SMS contient un lien direct permettant au client de finaliser sa commande sur une interface web.

1. **Reconnaissance du client (CRM léger)**

   * **Identification de l'appelant :** grâce au numéro appelant, l'IA reconnaît un client déjà venu et peut personnaliser l'accueil.

   * **Segmentation client connu / nouveau :** permet d'adapter les règles, par exemple proposer le paiement direct à non-habitué. (Pour les commandes)

1. **Interface de gestion (PC de caisse et mobile)**

   * **Dashboard de synchronisation :** interface centralisée pour visualiser instantanément le flux des réservations et commandes.

   * **Gestion simplifiée des arrivées et des retraits :** validation en un clic de l'arrivée d'un client ou du retrait d'une commande.

   * **Accès par rôle :** le gérant peut attribuer des accès limités à chaque employé.
   Consultation des transcriptions d'appels : chaque appel dispose d'une transcription consultable pour vérifier un échange.

1. **Supervision et hybridation (tableau de bord gérant)**

   * **Mise à jour dynamique du « cerveau » et gestion des menus :** pilotage de l'offre en temps réel (menus, plats en rupture).

   * **Suivi centralisé des ventes et commandes :** visibilité totale sur l'historique et le statut des transactions.

   * **Passerelle humaine (escalade d'appel) :** redirection transparente des appels complexes vers le personnel.

   * **Analyse de performance et ROI :** tableau de bord analytique pour visualiser le chiffre d'affaires préservé.

1. **Conformité légale et RGPD**

   * **Enregistrement légal des appels :** respect des règles de conservation et mention d'information en début d'appel.

   * **Gestion des litiges :** conservation des preuves (enregistrements/transcriptions) en cas de contestation.

1. **Architecture multi-tenant**

    La plateforme est conçue pour servir plusieurs établissements avec un cloisonnement strict des données.

**Stack technique**

* **Frontend :**

    * **Angular** pour le dashboard gérant/staff (application web riche, temps réel, gestion d'état complexe pour les réservations, commandes et transcriptions).

    * **Astro** pour le site vitrine et les pages marketing (rendu statique, SEO optimisé, temps de chargement minimal pour convertir les prospects commerçants).

* **Backend :**

    * **Python (FastAPI)** pour le cœur métier temps réel : orchestration de l'agent vocal, intégration STT/TTS/LLM, et WebSocket relié à la passerelle téléphonique (Twilio Media Streams). L'écosystème Python est incontournable sur l'IA — c'est le langage natif de `faster-whisper`, `coqui-tts` et `llama-cpp-python`, les trois briques que nous exécutons nous-mêmes. Architecture hexagonale (ports/adapters) pour que chaque modèle soit remplaçable sans toucher au domaine.

    * **Java (Spring Boot)** pour les services transactionnels critiques : paiements, gestion multi-tenant des réservations et commandes, facturation. La robustesse et la maturité de Java sont adaptées aux traitements où la fiabilité prime sur la latence brute.

* **Outils et services tiers :**

    * **Téléphonie : Twilio** (Programmable Voice + Media Streams) pour l'ingestion des appels en temps réel via WebSocket, la portabilité des numéros et la couverture internationale.

    * **STT (reconnaissance vocale) : faster-whisper `large-v3`**. C'est l'« oreille » du système : il transcrit en texte ce que dit le client au téléphone, en français et en direct. Auto-hébergé sur notre GPU : aucune donnée d'appel n'est envoyée à un tiers.

    * **TTS (synthèse vocale) : XTTS-v2** (Coqui). C'est la « voix » du système : il transforme les réponses écrites par l'IA en une voix naturelle, entendue par le client. Auto-hébergé sur notre GPU, avec une réponse quasi immédiate pour ne pas laisser de blanc dans la conversation.

    * **LLM : Qwen 2.5-14B-Instruct** (GGUF, quantisation Q4_K_M), exécuté **sur notre GPU** via `llama-cpp-python`. Auto-hébergé : aucune donnée d'appel n'est envoyée à un fournisseur tiers, et le coût est fixe (location GPU) plutôt qu'au token. Le modèle ne pilote pas le dialogue seul — le code tient l'état de la réservation (aiguilleur déterministe) et ne sollicite le modèle que pour comprendre le client et formuler les réponses, ce qui garantit qu'il ne « perd pas le fil » ni n'invente de réservation.

    * **Paiement :** Stripe

    * **Notifications :** Brevo (confirmations de réservation, rappels no-show, liens de paiement).

    * **Collaboration :** GitHub (code + CI/CD via GitHub Actions), Notion (documentation et suivi produit), Discord (communication d'équipe).

    * **Hébergement :** **Scaleway** (souveraineté européenne, conformité RGPD — différenciant vs Yumcall hébergé en Lettonie) pour le frontend et le backend ; **RunPod** (location de GPU) pour la partie IA, qui exécute STT, TTS et LLM auto-hébergés.

**Analyse des forces et faiblesses des choix technologiques**

| Choix | Forces | Faiblesses | Mitigation |
|---|---|---|---|
| **Angular (dashboard)** | Framework opinioné, TypeScript natif, RxJS adapté au temps réel (flux d'appels et de réservations), maintenabilité sur le long terme, expertise interne. | Courbe d'apprentissage plus raide que React/Vue, bundle plus lourd. | Lazy-loading par module, standalone components (Angular 17+). |
| **Astro (site vitrine)** | SEO excellent (SSG), performance maximale (zéro JS par défaut), coût d'hébergement faible. | Pas adapté aux applications interactives complexes — d'où la séparation avec Angular. | Astro reste cantonné aux pages publiques. |
| **Python (FastAPI)** | Écosystème IA dominant, async natif adapté au streaming audio, prototypage rapide pour l'agent vocal. | Performance CPU inférieure à la JVM, GIL sur les traitements CPU-bound. | Le CPU-bound est délégué aux API STT/TTS/LLM ; FastAPI ne fait que de l'I/O. |
| **Java (Spring Boot)** | Fiabilité éprouvée sur les transactions monétaires, écosystème mature (Spring Security, JPA), typage fort limitant les bugs en production. | Verbeux, temps de démarrage plus long, moins agile que Python pour itérer. | Cantonné aux services stables (paiement, facturation), peu itérés. |
| **Twilio** | Standard du marché, documentation exhaustive, Media Streams permettent le traitement audio en temps réel via WebSocket, portabilité des numéros et couverture internationale. | Coût à la minute élevé à grande échelle, dépendance à un acteur américain. | Renégociation tarifaire au volume ; abstraction du fournisseur derrière une interface pour migrer vers un opérateur européen (OVHcloud Telecom) en V2. |
| **faster-whisper (auto-hébergé)** | Aucune donnée envoyée à un tiers, coût fixe, transcription du français fiable. | Nécessite notre propre GPU ; qualité sensible au bruit de la ligne. | GPU dédié ; le client peut couper le bot à tout moment sans attendre. |
| **XTTS-v2 (auto-hébergé)** | Voix naturelle, réponse quasi immédiate, aucune donnée envoyée à un tiers. | Nécessite notre propre GPU. | GPU dédié ; moteur de voix remplaçable sans refonte. |
| **Qwen 2.5-14B (auto-hébergé)** | C'est le « cerveau » qui comprend ce que dit le client et formule des réponses naturelles ; auto-hébergé, donc aucune donnée d'appel envoyée à un tiers et coût fixe. | Un modèle laissé seul peut perdre le fil d'une réservation. | Le code garde la maîtrise de la réservation ; le modèle ne sert qu'à comprendre et à répondre. Il reste remplaçable sans refonte. |

**Alternatives d'architecture évaluées**

*1. Architecture applicative : monolithe modulaire vs micro-services*

| Approche | Forces | Faiblesses |
|---|---|---|
| **Monolithe modulaire (choisi)** | Déploiement simple, latence inter-modules nulle (critique pour l'agent vocal), debug facilité, coût d'infrastructure réduit, adapté à une équipe de 5 personnes. | Scaling vertical uniquement, couplage plus fort, risque de dette technique si les frontières entre modules ne sont pas respectées. |
| **Micro-services** | Scaling indépendant, résilience (une panne isolée), équipes autonomes. | Complexité opérationnelle (orchestration, observabilité), latence réseau inter-services incompatible avec l'objectif < 800 ms bout-en-bout sur l'agent vocal, overkill pour une équipe de 5 personnes en phase POC. |

**Décision (CTO) :** monolithe modulaire avec frontières internes strictes (bounded contexts : Voice, Reservation, Order, Payment, Tenant). Migration vers des micro-services envisagée uniquement si un module devient un goulot d'étranglement mesuré (> 1 000 établissements actifs).

*2. Interface staff/gérant : PWA vs application native*

| Approche | Forces | Faiblesses |
|---|---|---|
| **PWA responsive (choisie)** | Une seule base de code (Angular), déploiement instantané sans store, fonctionne sur le PC de caisse ET les téléphones secondaires (exigence explicite du restaurateur), installation possible sur écran d'accueil. | Notifications push moins fiables sur iOS, accès matériel restreint (pas critique ici). |
| **Application native (iOS + Android)** | Meilleures notifications push, accès complet au matériel, ressenti "premium". | Deux bases de code supplémentaires, cycles de validation sur les stores, effort disproportionné au regard des besoins réels (dashboard + notifications simples). |

**Décision (CTO) :** PWA. Le besoin métier identifié en interview est "centralisé sur le PC de caisse + responsive sur téléphone secondaire" — la PWA couvre exactement ce périmètre sans doubler l'effort de développement.

*3. Agent vocal : pipeline STT → LLM → TTS vs modèle vocal end-to-end (ex. GPT-4o Realtime, Gemini Live)*

| Approche | Forces | Faiblesses |
|---|---|---|
| **Pipeline modulaire (choisi)** | Chaque brique remplaçable indépendamment, coût maîtrisé, transcription réutilisable pour la conformité légale et la consultation dans le dashboard, choix du meilleur fournisseur pour chaque étape. | Latence cumulée à optimiser, orchestration plus complexe. |
| **Modèle vocal end-to-end** | Latence naturellement plus faible (~500 ms), prosodie et interruptions gérées nativement. | Coût très élevé, dépendance forte à un unique fournisseur, transcription textuelle moins précise, moins de contrôle sur les tool calls (indispensables pour interroger agenda et menu). |

**Décision (CTO) :** pipeline modulaire pour le POC. Les modèles end-to-end seront réévalués en V2, lorsque leurs prix auront baissé et que le tool use temps réel sera mature. Cette décision est motivée par l'exigence de contrôle fin (fallback, escalade humaine, transcription légale) exprimée par le restaurateur interviewé.

*4. Multi-tenant : base partagée vs base par tenant*

| Approche | Forces | Faiblesses |
|---|---|---|
| **Base partagée avec `tenant_id` (choisie)** | Coût d'infrastructure faible, maintenance et migrations simplifiées, analytics cross-tenant possibles. | Risque de fuite de données si le cloisonnement applicatif est mal implémenté. |
| **Base par tenant** | Isolation forte, conformité RGPD facilitée, sauvegarde/restauration par client. | Coût opérationnel élevé, migrations complexes, difficile à scaler au-delà de 50 clients. |

**Décision (CTO) :** base partagée avec `tenant_id` obligatoire sur chaque table, filtres appliqués via Row-Level Security PostgreSQL et middleware applicatif. Audit de sécurité systématique sur chaque nouvelle requête.

**Synthèse de la décision technique**

Le fil rouge de nos choix est double :

1. **La latence perçue par le client final** — priorité n°1 identifiée en interview ("prendre un appel ne doit pas prendre 4 ans") — qui justifie le monolithe, le pipeline vocal optimisé et Twilio Media Streams.

2. **La souveraineté et la maîtrise des coûts** — l'auto-hébergement des trois modèles (STT, TTS, LLM) sur GPU garantit qu'aucune donnée d'appel ne sort vers une API tierce, transforme un coût au token/à la minute en coût fixe, et rend chaque brique remplaçable via une interface d'abstraction. Le contrepoint assumé : une charge d'exploitation GPU à opérer.

Chaque choix est réversible via des interfaces d'abstraction (adaptateur LLM, adaptateur téléphonie), afin que les décisions prises aujourd'hui ne bloquent pas les évolutions futures.

## Plan d'action : Vision / Objectifs / KPIs

**Vision**

Devenir l'assistant vocal de référence des commerces de proximité en France : ne plus jamais laisser un appel sans réponse, tout en préservant la relation humaine et la souveraineté des données (auto-hébergement des modèles IA, hébergement européen). Le POC se concentre sur la prise de réservation en restauration ; les commandes à emporter et les autres métiers (kiné, garagiste) constituent la V2.

**Objectifs & KPIs**

| Objectif | KPI cible (POC) |
|---|---|
| Ne rater aucun appel | 100 % des appels décrochés par l'IA |
| Réactivité perçue « zéro latence » | < 800 ms entre la fin de la phrase du client et le début de la réponse |
| Fiabilité de la compréhension | > 90 % de réservations sans erreur de date/nombre/nom |
| Naturalité (pas d'effet robot) | Note de naturalité ≥ 4/5 sur un panel de test |
| Fiabilité de l'enregistrement | 100 % des réservations confirmées visibles au dashboard en temps réel |
| Filet de sécurité | 0 appel terminé en silence (rappel garanti si échec) |

**Profils de l'équipe (5 personnes)**

| Membre | Rôle | Compétences principales |
|---|---|---|
| Rayane Mabrouki | Ingénieur IA & Voix | Agent vocal (STT/TTS/LLM), Python, orchestration temps réel, déploiement GPU |
| Jugurta Mahious | Développeur Backend (+ appui front) | Java/Spring Boot, API, réservations, base de données, multi-tenant |
| Hassan Aidibe | Développeur Frontend (+ appui back) | Angular (dashboard temps réel), UX, coordination du backlog (Notion) |
| Victor Huang | Développeur Frontend & Paiement (+ appui back) | Angular, intégration Stripe, parcours de paiement |
| Vitomir Laces | Développeur Frontend & Téléphonie (+ appui back) | Angular, intégration Twilio (flux de l'appel), site vitrine |

**Répartition des tâches**

* **Agent vocal (IA & Voix)** — pipeline reconnaissance → compréhension → voix, gestion des interruptions, filet de sécurité : *Rayane*.
* **Services transactionnels (Backend)** — réservations, disponibilités, multi-tenant, base de données : *Jugurta*, avec un appui ponctuel côté front.
* **Téléphonie (flux de l'appel)** — intégration Twilio, arrivée et acheminement des appels : *Vitomir*.
* **Paiement** — intégration Stripe, parcours et sécurisation du règlement : *Victor*.
* **Interfaces (Frontend)** — dashboard gérant/staff, plan de salle temps réel, site vitrine : *Hassan*, *Victor* et *Vitomir* (chacun contribuant aussi un peu au backend).
* **Coordination** — backlog Notion, suivi Kanban et préparation de la soutenance : assurée collectivement, pilotée par *Hassan*.

**Planning provisoire**

> Les dates des phases déjà réalisées sont indicatives — à ajuster par l'équipe.

| Phase | Période | État | Livrable |
|---|---|---|---|
| 0 — Cadrage & veille | Printemps 2026 | ✅ Fait | Ce rapport de cadrage, interviews, architecture cible |
| 1 — POC : prise de réservation | Été 2026 | Démo : un appel réel aboutit à une réservation enregistrée (dashboard temps réel, plan de salle, filet de sécurité, barge-in), présentée à la **soutenance de rentrée** |
| ★ Soutenance | Rentrée 2026 | À venir | Présentation du POC au jury |
| 2 — Enrichissement | Automne 2026 → | À venir | Commande à emporter + paiement, CRM léger, support multilingue |
| 3 — Industrialisation | Fin 2026 → | À venir | Multi-tenant à l'échelle, conformité RGPD, supervision / monitoring, SLA |

**Estimation des coûts**

*Ressources humaines (masse salariale simulée, coût employeur ≈ 1,42 × brut) :*

| Rôle | Brut annuel | Coût employeur annuel |
|---|---|---|
| Ingénieur IA & Voix | 50 000 € | ~71 000 € |
| Développeur Backend | 45 000 € | ~63 900 € |
| Développeur Frontend & Téléphonie | 44 000 € | ~62 480 € |
| Développeur Frontend & Paiement | 43 000 € | ~61 060 € |
| Développeur Frontend & Coordination | 43 000 € | ~61 060 € |
| **Total** | **225 000 €** | **~319 500 €/an** (~26 600 €/mois) |

*Matériel (investissement initial) :* 5 postes de développement (~1 500 € × 5 = 7 500 €), périphériques et écrans (~2 000 €), amortis sur 3 ans.

*Infrastructure (mensuel estimé, hors volume d'appels) :*

| Poste | Coût mensuel |
|---|---|
| GPU IA (RunPod / hébergeur GPU européen en prod) | ~180 € |
| Hébergement front + back + base PostgreSQL (Scaleway) | ~70 € |
| Téléphonie (Twilio : numéro + minutes de test) | ~30 € |
| Notifications SMS/e-mail (Brevo) | ~20 € |
| Nom de domaine, outils divers | ~15 € |
| **Total infrastructure** | **~315 €/mois** |

*Fonctionnement & maintenance :* outils collaboratifs (Notion, GitHub, Discord — offres gratuites ou faible coût), supervision et monitoring (~30 €/mois), correctifs et mises à jour intégrés au temps de l'équipe.

*SLA & support :* en phase commerciale, une astreinte technique et un support client de niveau 1 seront provisionnés (objectif de disponibilité 99,5 %). En phase POC, le support est assuré par l'équipe sur les heures ouvrées.

## Gestion de projet

Bienvenue dans l'équipe ! Voici comment nous travaillons au quotidien.

**Nos outils**

* **Notion** : gestion des epics, user stories, tâches et documentation produit.
* **GitHub** : code source, pull requests, CI/CD (GitHub Actions).
* **Discord** : communication d'équipe, notifications de PR, alertes CI.

**Méthodologie**

Nous fonctionnons en **Kanban** (flux continu, pas de sprints fixes) avec :

* Un **daily meeting** pour se synchroniser.
* Une **rétrospective** à la fin de chaque epic pour identifier les axes d'amélioration.

**1. Créer un ticket dans Notion**

Rends-toi dans *QG de l'espace de Hassan Aidibe → Projet Alloquence → Epics*.

Une **epic** se structure en trois blocs :

* **Objectif** — ex : *Transformer un appel en réservation correctement enregistrée.*
* **Valeur** — ex : *L'action concrète qui génère du chiffre d'affaires côté restaurateur.*
* **Couvre les User Stories** — ex : *US 2.1 à US 2.4 (prise de réservation, prise de commande à emporter V2, vérification de disponibilité, gestion des alternatives).*

Rattache ensuite une **user story** à l'epic :

> **En tant que** client appelant, **je veux** réserver une table en donnant date, heure, nombre de couverts et mon nom, afin d'avoir ma table sans attendre.
>
> **Critères d'acceptation :**
> - Le bot collecte date, heure, couverts, nom, numéro.
> - La réservation est enregistrée et vérifiable côté restaurateur.
> - Les noms et dates sont reconfirmés à l'oral et par SMS.

Puis découpe-la en **tâches** exécutables, par exemple :

* Tâche 1 : Modéliser l'entité Réservation.
* Tâche 2 : Enregistrer la réservation et la rendre visible sur le dashboard.
* Tâche 3 : Implémenter le dialogue de collecte des informations.

**2. Créer sa branche**

Grâce à l'intégration Notion ↔ GitHub, le nom de branche est généré automatiquement depuis le ticket (préfixe `feat/`, `fix/` ou `chore/` + référence).

Sur le repo concerné (ex : `backend`), crée toujours ta branche **à partir de `staging`**.

**3. Commits**

Nous suivons la [Convention Commits](https://www.conventionalcommits.org/fr/v1.0.0/).

Exemple : `feat: ajout du dialogue de collecte de réservation`.

**4. Demande de fusion (Pull Request)**

* Cible : la branche **`staging`**.
* **1 reviewer minimum** obligatoire.
* **CI 100 % verte** avant merge.
* Stratégie de merge : **squash** (historique propre).
* Notifie ton binôme sur Discord et mets le ticket Notion à jour.

**5. Mise en production**

Si le reviewer valide et que la CI est verte, le code part sur `staging`. La promotion vers `main` (production) se fait ensuite selon le même circuit de revue.

**6. Captures d'écran**

![alt text](image-6.png)

![alt text](image-7.png)

![alt text](image-8.png)

![alt text](image-9.png)

![alt text](image-10.png)

![alt text](image-11.png)

## La solution (POC)

Le POC réalise le parcours central : **un appel téléphonique réel aboutit à une réservation enregistrée et visible en temps réel** sur le dashboard du restaurateur.

**Fonctionnalités démontrées (captures d'écran)**

* **Liste du jour** : les réservations du jour, dont celles prises automatiquement par l'assistant vocal (badge « Bot »), avec les chiffres clés en tête (couverts, réservations captées par le bot, taux de confirmation). Le staff filtre par statut, confirme, annule ou place chaque réservation, et reçoit une notification en direct à chaque nouvelle réservation captée par le bot.

![alt text](Liste_du_jour.png)

* **Tableau de bord** (« Vue d'ensemble du service de ce soir ») : quatre indicateurs clés en tête — nombre de réservations, couverts attendus, **réservations captées par le bot**, taux de confirmation. En dessous, un graphique d'**affluence du service** (couverts attendus par créneau horaire), une répartition **agent vocal vs saisie manuelle** qui met en évidence la part prise par l'assistant, et un aperçu des prochaines réservations.

![alt text](Tableau_de_bord.png)

* **Plan de salle** : carte visuelle des tables (vue 2D et 3D), chaque table prenant automatiquement son statut à partir des réservations — **Libre**, **Réservée** ou **Installée**, avec un signalement de retard (« +25 min »). Le staff place les réservations en attente sur les tables (la meilleure table libre est suggérée, avec fusion possible pour les grands groupes), dispose d'un **mode service** plein écran pour l'accueil et d'une **simulation de soirée** qui projette l'occupation heure par heure sans rien modifier. En temps réel, une réservation prise par le bot fait *clignoter* sa table : on voit l'assistant travailler.

![alt text](Plan_de_salle.png)

**Schéma de déploiement**

```
        Appel téléphonique (client)
                  │
                  ▼
        ┌───────────────────┐
        │      Twilio        │  Passerelle voix (Media Streams / WebSocket)
        └─────────┬─────────┘
                  │  flux audio temps réel
                  ▼
  ┌───────────────────────────────────────────┐
  │        Service Agent Vocal (Python)        │
  │  Orchestration STT → LLM → TTS + barge-in  │
  └───┬───────────────┬───────────────┬────────┘
      │               │               │
      ▼               ▼               ▼
   [ STT ]         [ LLM ]         [ TTS ]         ← RunPod (GPU) : modèles auto-hébergés
 faster-whisper   Qwen 2.5-14B     XTTS-v2
      │
      ▼  (réservation validée)
  ┌───────────────────────────┐
  │   Backend (Java / Spring)  │  ← Scaleway (Europe)
  │  Réservations, multi-tenant│
  │  PostgreSQL                │
  └─────────────┬─────────────┘
                │  temps réel
                ▼
  ┌───────────────────────────┐
  │   Dashboard (Angular)      │  ← Scaleway (Europe)
  │   Gérant / staff           │
  └───────────────────────────┘
```

**Justification de l'infrastructure au regard des objectifs SLA**

* **Performance / latence** — priorité n°1 du restaurateur. Les modèles IA sont **auto-hébergés sur GPU** (RunPod) : pas d'aller-retour vers une API distante à chaque tour de parole. Le TTS diffuse dès le premier morceau audio (streaming) et le barge-in laisse le client couper le bot, ce qui supprime les silences perçus.
* **Disponibilité** — front, back et base sont hébergés sur Scaleway avec redémarrage automatique ; l'objectif de production est de 99,5 %. Twilio apporte une couverture opérateur de niveau industriel pour l'arrivée des appels.
* **Sécurité** — hébergement européen (RGPD), cloisonnement multi-tenant strict (`tenant_id` sur chaque donnée), et **aucune donnée d'appel envoyée à une API d'IA tierce** puisque les modèles tournent chez nous.
* **Résilience** — architecture en briques remplaçables (interfaces d'abstraction téléphonie / STT / TTS / LLM) : la panne ou l'indisponibilité d'un fournisseur n'impose pas de refonte. Le filet de sécurité garantit qu'aucun appel ne se termine en silence.
* **Gestion des pics** (samedi soir, heures de rush) — le backend transactionnel (Java/Spring) est dimensionné pour la charge ; la partie IA peut monter en puissance en ajoutant des GPU derrière le service d'orchestration.

**Risques opérationnels et atténuation**

| Risque | Atténuation |
|---|---|
| GPU unique = point de défaillance de la partie IA | Basculement prévu vers un second GPU ; interfaces d'abstraction permettant un repli temporaire sur une API SaaS. |
| Latence dégradée aux heures de pointe | Modèles auto-hébergés (pas de file d'attente d'API tierce) ; ajout de GPU à la demande. |
| Coupure réseau entre la téléphonie et le service IA | Supervision et redémarrage automatique ; filet de sécurité (rappel garanti) si un appel ne peut aboutir. |
| Erreur de transcription (bruit de ligne, nom mal compris) | Reconfirmation systématique des noms et dates à l'oral **et** par SMS. |
| Dépendance à un fournisseur (téléphonie) | Interface d'abstraction téléphonie pour migrer vers un opérateur européen sans réécrire le cœur. |
| Conformité RGPD | Hébergement européen, données maîtrisées de bout en bout, information légale en début d'appel. |

*Cette section restera volontairement moins détaillée que le reste du rapport, le POC étant une démonstration ciblée du parcours de réservation.*
