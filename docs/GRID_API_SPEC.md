# GRID API spec (extracted from gridapi.docx)

Query the SeriesState of the latest series that this player, given by its ID, has participated in. Returns SeriesState.
Argument
Type
Description
id
ID!
None
Objects
AbilityValorant
An ability that is owned by a player. (available from version "1.8")
Field
Type
Description
id
ID!
GRID ID for this ability.
name
String!
The name of this ability.
ready
Boolean!
Whether this ability can be activated or not.
charges
Int!
The amount of charges this ability has.(available from version "3.21")
Bounds
Bounds information (ie min and max Coordinates for a map).
Field
Type
Description
min
Coordinates!
Minimum Coordinates value.
max
Coordinates!
Maximum Coordinates value.
Character
In-game character (ie champion, class) information
Field
Type
Description
id
ID!
GRID Character ID.
name
String!
Character name
ClockState
The game or respawn clock state. (available from version "1.3")
Field
Type
Description
id
String
The id of this clock
type
String
The type of this clock
ticking
Boolean
Indicates if this clock is ticking
ticksBackwards
Boolean
Indicates if this clock is ticking backwards
currentSeconds
Int
The current seconds for this clock
Coordinates
Spatial Coordinates.
Field
Type
Description
x
Float!
X coordinate value.
y
Float!
Y coordinate value.
DamageDealtSource
A source of damage with the amount and occurrence. (available from version "3.17")
Field
Type
Description
id
ID!
GRID ID for this damage source
source
DamageSource!
Source of the damage.
damageAmount
Int!
Amount of damage dealt with the source.
occurrenceCount
Int!
Amount of times damage was dealt with the source.
damageTypes
[DamageDealtType!]!
Breakdown of different damage types.(available from version "3.18")
DamageDealtTarget
A target of damage with the amount and occurrence. (available from version "3.17")
Field
Type
Description
id
ID!
GRID ID for this damage target
target
DamageTarget!
Target of the damage.
damageAmount
Int!
Amount of damage dealt to the target.
occurrenceCount
Int!
Amount of times damage was dealt to the target.
damageTypes
[DamageDealtType!]!
Breakdown of different damage types.(available from version "3.18")
DamageDealtType
A type of dealt damage. (available from version "3.18")
Field
Type
Description
id
ID!
GRID ID for this damage target
type
String!
Name of the type of damage
damageAmount
Int!
Amount of damage dealt.
occurrenceCount
Int!
Amount of times damage was dealt.
DamageSource
A source of damage with a name. (available from version "3.17")
Field
Type
Description
name
String!
Name of the damage source.
DamageTarget
A target of damage with name. (available from version "3.17")
Field
Type
Description
name
String!
Name of the damage source.
DataProvider
A data provider which provides external entity IDs.
Field
Type
Description
name
String!
Unique name of the data provider
DefaultAbility
An ability that is owned by a player. (available from version "1.8")
Field
Type
Description
id
ID!
GRID ID for this ability.
name
String!
The name of this ability.
ready
Boolean!
Whether this ability can be activated or not.
DraftAction
A draft action occurrence such as a team banning a map or a player picking a champion.
Field
Type
Description
id
ID!
GRID draft action ID.
type
String!
Type of the draft action.
sequenceNumber
String!
Sequence number of the draft action.
drafter
Drafter!
Entity performing the draft action.
draftable
Draftable!
Entity being drafted.
Draftable
Entity being drafted.
Field
Type
Description
id
ID!
GRID ID of the entity being drafted.
type
String!
Type of the entity being drafted.
name
String!
Name of the entity being drafted.
linkedDraftable
[Draftable!]
List of related draftables.(available from version "3.32")
Drafter
Entity performing a draft action.
Field
Type
Description
id
ID!
GRID ID of the entity performing a draft action.
type
String!
Type of entity performing a draft action.
ExternalEntity
An external entity representing the entity it is embedded in.
Field
Type
Description
id
ID!
ID representing this entity
ExternalLink
A link to an external entity given via an ID. (available from version "1.2")
Field
Type
Description
dataProvider
DataProvider!
A data provider which provides external entity IDs.
externalEntity
ExternalEntity!
An external entity representing the entity it is embedded in.
GamePlayerStateCs2
CS2 data points for a Player, aggregated for a Game.
Field
Type
Description
id
ID!
GRID ID of this Player.
name
String!
Name of this Player.
character
Character
In-game character (ie champion, class) of the Player in this Game
roles
[PlayerRole!]!
Roles this Player has in this Game.(available from version "3.43")
participationStatus
ParticipationStatus!
Indicates the participation status of a player for this Game
money
Int!
Amount of money this Player is holding in cash.Depending on the Title this may represent in-game gold, credits, gems etc.
loadoutValue
Int!
Total value (worth in money) of this Player’s loadout.
netWorth
Int!
Sum of money and inventoryValue for this Player.
kills
Int!
Number of enemy kills for this Player in this Game.
killAssistsReceived
Int!
Number of enemy kill assists received by this Player in this Game.
killAssistsGiven
Int!
Number of enemy kill assists provided by this Player in this Game.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists received by this Player in this Game.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this Player in this Game.
teamkills
Int!
Number of teamkills for this Player in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this Player in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this Player in this Game.A teamkill is the occurrence of killing an allied member
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists received by this Player in this Game.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of teamkills, which were executed with a specific weapon by this Player in this Game.A teamkill is the occurrence of killing an allied member.
selfkills
Int!
Number of selfkills (suicides) for this Player in this Game.
deaths
Int!
Number of deaths for this Player in this Game.
firstKill
Boolean!
Indication of player acquiring first kill.(available from version "3.10")
structuresDestroyed
Int!
Number of structures destroyed by this Player in this Game.
structuresCaptured
Int!
Number of structures captured by this Player in this Game.
inventory
PlayerInventory!
The Player's inventory
objectives
[Objective!]!
A list of objectives that were fulfilled by this Player in this Game.
position
Coordinates
Player Coordinates on the map
multikills
[Multikill!]!
A list of multi kills that happened by this Player in this Game.
unitKills
[UnitKill!]!
A list of unit kills that happened by this Player in this Game.(available from version "3.1")
abilities
[DefaultAbility!]!
A list of abilities that are owned by this Player in this Game.
statusEffects
[StatusEffect!]!
A list of status effects affecting the player(available from version "3.31")
alive
Boolean!
Indicates whether the player is alive.
currentArmor
Int!
The amount of current armor.
currentHealth
Int!
The current health of the player.
damageDealt
Int!
The amount of damage dealt.
damageTaken
Int!
The amount of total damage taken.
maxHealth
Int!
The max amount of health of the player.
selfdamageDealt
Int!
The amount of damage dealt to self.
selfdamageTaken
Int!
The amount of damage taken from self.
teamdamageDealt
Int!
The amount of damage dealt to team.
teamdamageTaken
Int!
The amount of damage taken from team.
headshots
Int!
Number of enemy headshots for this Player.
teamHeadshots
Int!
Number of allied headshots for this Player.
externalLinks
[ExternalLink!]!
A list of ExternalLink, containing information about links to externally provided IDs.
damageDealtSources
[DamageDealtSource!]!
A list of damage sources.(available from version "3.17")
damageDealtTargets
[DamageDealtTarget!]!
A list of damage targets.(available from version "3.17")
GamePlayerStateCsgo
CSGO data points for a Player, aggregated for a Game. (available from version "1.1")
Field
Type
Description
id
ID!
GRID ID of this Player.
name
String!
Name of this Player.
character
Character
In-game character (ie champion, class) of the Player in this Game
roles
[PlayerRole!]!
Roles this Player has in this Game.(available from version "3.43")
participationStatus
ParticipationStatus!
Indicates the participation status of a player for this Game
money
Int!
Amount of money this Player is holding in cash.Depending on the Title this may represent in-game gold, credits, gems etc.
loadoutValue
Int!
Total value (worth in money) of this Player’s loadout.
netWorth
Int!
Sum of money and inventoryValue for this Player.
kills
Int!
Number of enemy kills for this Player in this Game.
killAssistsReceived
Int!
Number of enemy kill assists received by this Player in this Game.
killAssistsGiven
Int!
Number of enemy kill assists provided by this Player in this Game.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists received by this Player in this Game.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this Player in this Game.
teamkills
Int!
Number of teamkills for this Player in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this Player in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this Player in this Game.A teamkill is the occurrence of killing an allied member
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists received by this Player in this Game.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of teamkills, which were executed with a specific weapon by this Player in this Game.A teamkill is the occurrence of killing an allied member.
selfkills
Int!
Number of selfkills (suicides) for this Player in this Game.
deaths
Int!
Number of deaths for this Player in this Game.
firstKill
Boolean!
Indication of player acquiring first kill.(available from version "3.10")
structuresDestroyed
Int!
Number of structures destroyed by this Player in this Game.
structuresCaptured
Int!
Number of structures captured by this Player in this Game.
inventory
PlayerInventory!
The Player's inventory
objectives
[Objective!]!
A list of objectives that were fulfilled by this Player in this Game.
position
Coordinates
Player Coordinates on the map
multikills
[Multikill!]!
A list of multi kills that happened by this Player in this Game.(available from version "1.7")
unitKills
[UnitKill!]!
A list of unit kills that happened by this Player in this Game.(available from version "3.1")
abilities
[DefaultAbility!]!
A list of abilities that are owned by this Player in this Game.(available from version "1.8")
statusEffects
[StatusEffect!]!
A list of status effects affecting the player(available from version "3.31")
alive
Boolean!
Indicates whether the player is alive.(available from version "1.5")
currentArmor
Int!
The amount of current armor.
currentHealth
Int!
The current health of the player.(available from version "1.5")
damageDealt
Int!
The amount of damage dealt.
damageTaken
Int!
The amount of total damage taken.
maxHealth
Int!
The max amount of health of the player.(available from version "1.5")
selfdamageDealt
Int!
The amount of damage dealt to self.
selfdamageTaken
Int!
The amount of damage taken from self.
teamdamageDealt
Int!
The amount of damage dealt to team.
teamdamageTaken
Int!
The amount of damage taken from team.
headshots
Int!
Number of enemy headshots for this Player.
teamHeadshots
Int!
Number of allied headshots for this Player.
externalLinks
[ExternalLink!]!
A list of ExternalLink, containing information about links to externally provided IDs.(available from version "2.2")
damageDealtSources
[DamageDealtSource!]!
A list of damage sources.(available from version "3.17")
damageDealtTargets
[DamageDealtTarget!]!
A list of damage targets.(available from version "3.17")
GamePlayerStateDefault
Default data points for a Player, aggregated for a Game
Field
Type
Description
id
ID!
GRID ID of this Player.
name
String!
Name of this Player.
character
Character
In-game character (ie champion, class) of the Player in this Game
roles
[PlayerRole!]!
Roles this Player has in this Game.(available from version "3.43")
participationStatus
ParticipationStatus!
Indicates the participation status of a player for this Game
money
Int!
Amount of money this Player is holding in cash.Depending on the Title this may represent in-game gold, credits, gems etc.
loadoutValue
Int!
Total value (worth in money) of this Player’s loadout.
netWorth
Int!
Sum of money and inventoryValue for this Player.
kills
Int!
Number of enemy kills for this Player in this Game.
killAssistsReceived
Int!
Number of enemy kill assists received by this Player in this Game.
killAssistsGiven
Int!
Number of enemy kill assists provided by this Player in this Game.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists received by this Player in this Game.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this Player in this Game.
headshots
Int!
Number of enemy headshots for this Player.(available until version "1.9")
teamkills
Int!
Number of teamkills for this Player in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this Player in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this Player in this Game.A teamkill is the occurrence of killing an allied member
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists received by this Player in this Game.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of teamkills, which were executed with a specific weapon by this Player in this Game.A teamkill is the occurrence of killing an allied member.
teamHeadshots
Int!
Number of allied headshots for this Player.(available until version "1.9")
selfkills
Int!
Number of selfkills (suicides) for this Player in this Game.
deaths
Int!
Number of deaths for this Player in this Game.
firstKill
Boolean!
Indication of player acquiring first kill.(available from version "3.10")
structuresDestroyed
Int!
Number of structures destroyed by this Player in this Game.
structuresCaptured
Int!
Number of structures captured by this Player in this Game.
inventory
PlayerInventory!
The Player's inventory
objectives
[Objective!]!
A list of objectives that were fulfilled by this Player in this Game.
position
Coordinates
Player Coordinates on the map
multikills
[Multikill!]!
A list of multi kills that happened by this Player in this Game.(available from version "1.7")
unitKills
[UnitKill!]!
A list of unit kills that happened by this Player in this Game.(available from version "3.1")
abilities
[DefaultAbility!]!
A list of abilities that are owned by this Player in this Game.(available from version "1.8")
statusEffects
[StatusEffect!]!
A list of status effects affecting the player(available from version "3.31")
GamePlayerStateDota
Dota data points for a Player, aggregated for a Game. (available from version "1.1")
Field
Type
Description
id
ID!
GRID ID of this Player.
name
String!
Name of this Player.
character
Character
In-game character (ie champion, class) of the Player in this Game
roles
[PlayerRole!]!
Roles this Player has in this Game.(available from version "3.43")
participationStatus
ParticipationStatus!
Indicates the participation status of a player for this Game
money
Int!
Amount of money this Player is holding in cash.Depending on the Title this may represent in-game gold, credits, gems etc.
loadoutValue
Int!
Total value (worth in money) of this Player’s loadout.
netWorth
Int!
Sum of money and inventoryValue for this Player.
kills
Int!
Number of enemy kills for this Player in this Game.
killAssistsReceived
Int!
Number of enemy kill assists received by this Player in this Game.
killAssistsGiven
Int!
Number of enemy kill assists provided by this Player in this Game.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists received by this Player in this Game.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this Player in this Game.
headshots
Int!
Number of enemy headshots for this Player.(available until version "1.9")
teamkills
Int!
Number of teamkills for this Player in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this Player in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this Player in this Game.A teamkill is the occurrence of killing an allied member
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists received by this Player in this Game.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of teamkills, which were executed with a specific weapon by this Player in this Game.A teamkill is the occurrence of killing an allied member.
teamHeadshots
Int!
Number of allied headshots for this Player.(available until version "1.9")
selfkills
Int!
Number of selfkills (suicides) for this Player in this Game.
deaths
Int!
Number of deaths for this Player in this Game.
firstKill
Boolean!
Indication of player acquiring first kill.(available from version "3.10")
structuresDestroyed
Int!
Number of structures destroyed by this Player in this Game.
structuresCaptured
Int!
Number of structures captured by this Player in this Game.
inventory
PlayerInventory!
The Player's inventory
objectives
[Objective!]!
A list of objectives that were fulfilled by this Player in this Game.
position
Coordinates
Player Coordinates on the map
multikills
[Multikill!]!
A list of multi kills that happened by this Player in this Game.(available from version "1.7")
unitKills
[UnitKill!]!
A list of unit kills that happened by this Player in this Game.(available from version "3.1")
abilities
[DefaultAbility!]!
A list of abilities that are owned by this Player in this Game.(available from version "1.8")
statusEffects
[StatusEffect!]!
A list of status effects affecting the player(available from version "3.31")
alive
Boolean!
Indicates whether the player is alive.
currentHealth
Int!
The current health of the player.(available from version "1.6")
experiencePoints
Int!
The amount of experience points gathered by this player.
maxHealth
Int!
The max amount of health of the player.(available from version "1.6")
externalLinks
[ExternalLink!]!
A list of ExternalLink, containing information about links to externally provided IDs.(available from version "3.6")
killStreak
Int!
The current kill streak of the player(available from version "3.44")
GamePlayerStateLol
LoL data points for a Player, aggregated for a Game.
Field
Type
Description
id
ID!
GRID ID of this Player.
name
String!
Name of this Player.
character
Character
In-game character (ie champion, class) of the Player in this Game
roles
[PlayerRole!]!
Roles this Player has in this Game.(available from version "3.43")
participationStatus
ParticipationStatus!
Indicates the participation status of a player for this Game
money
Int!
Amount of money this Player is holding in cash.Depending on the Title this may represent in-game gold, credits, gems etc.
loadoutValue
Int!
Total value (worth in money) of this Player’s loadout.
netWorth
Int!
Sum of money and inventoryValue for this Player.
kills
Int!
Number of enemy kills for this Player in this Game.
killAssistsReceived
Int!
Number of enemy kill assists received by this Player in this Game.
killAssistsGiven
Int!
Number of enemy kill assists provided by this Player in this Game.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists received by this Player in this Game.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this Player in this Game.
headshots
Int!
Number of enemy headshots for this Player.(available until version "1.9")
teamkills
Int!
Number of teamkills for this Player in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this Player in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this Player in this Game.A teamkill is the occurrence of killing an allied member
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists received by this Player in this Game.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of teamkills, which were executed with a specific weapon by this Player in this Game.A teamkill is the occurrence of killing an allied member.
teamHeadshots
Int!
Number of allied headshots for this Player.(available until version "1.9")
selfkills
Int!
Number of selfkills (suicides) for this Player in this Game.
deaths
Int!
Number of deaths for this Player in this Game.
firstKill
Boolean!
Indication of player acquiring first kill.(available from version "3.10")
structuresDestroyed
Int!
Number of structures destroyed by this Player in this Game.
structuresCaptured
Int!
Number of structures captured by this Player in this Game.
inventory
PlayerInventory!
The Player's inventory
objectives
[Objective!]!
A list of objectives that were fulfilled by this Player in this Game.
position
Coordinates
Player Coordinates on the map
multikills
[Multikill!]!
A list of multi kills that happened by this Player in this Game.(available from version "1.7")
unitKills
[UnitKill!]!
A list of unit kills that happened by this Player in this Game.(available from version "3.1")
abilities
[DefaultAbility!]!
A list of abilities that are owned by this Player in this Game.(available from version "1.8")
statusEffects
[StatusEffect!]!
A list of status effects affecting the player(available from version "3.31")
alive
Boolean!
Indicates whether the player is alive.
currentArmor
Int!
The amount of current armor.
currentHealth
Int!
The current health of the player.
damageDealt
Int!
The amount of damage dealt.(available from version "3.23")
damagePercentage
Float!
The percentage of damage dealt compared to the overall damage dealt by the player's team.(available from version "3.25")
damagePerMinute
Float!
The amount of damage dealt per minute.(available from version "3.24")
damagePerMoney
Float!
The amount of damage dealt per money earned.(available from version "3.26")
damageTaken
Int!
The amount of damage taken.(available from version "3.23")
experiencePoints
Int!
The amount of experience points gathered by this player.
forwardPercentage
Float!
The percentage of game time this player spent on the opponent's side of the map.(available from version "3.42")
kdaRatio
Float!
The ratio of kills and assists given vs deaths.(available from version "3.27")
killParticipation
Float!
The percentage of this player's kills and assists compared to the overall team's kills.(available from version "3.35")
killsAndAssists
Float!
The sum of kills and assists given.(available from version "3.34")
maxHealth
Int!
The max amount of health of the player.
moneyPercentage
Float!
The percentage of money earned compared to the overall money earned by the player's team.(available from version "3.37")
moneyPerMinute
Float!
The amount of money earned per minute.(available from version "3.36")
respawnClock
ClockState!
Respawn clock state to indicate when this player respawns.(available from version "3.3")
totalMoneyEarned
Int!
The total amount of money that was earned by this player.(available from version "3.2")
visionScore
Float!
Indicates how much vision this player has influenced in the game, including the vision that was granted and denied.(available from version "3.30")
visionScorePerMinute
Float!
The vision score gained per minute.(available from version "3.33")
GamePlayerStateMlbb
MLBB data points for a Player, aggregated for a Game
Field
Type
Description
id
ID!
GRID ID of this Player.
name
String!
Name of this Player.
character
Character
In-game character (ie champion, class) of the Player in this Game
roles
[PlayerRole!]!
Roles this Player has in this Game.
participationStatus
ParticipationStatus!
Indicates the participation status of a player for this Game
money
Int!
Amount of money this Player is holding in cash.Depending on the Title this may represent in-game gold, credits, gems etc.
loadoutValue
Int!
Total value (worth in money) of this Player’s loadout.
netWorth
Int!
Sum of money and inventoryValue for this Player.
kills
Int!
Number of enemy kills for this Player in this Game.
killAssistsReceived
Int!
Number of enemy kill assists received by this Player in this Game.
killAssistsGiven
Int!
Number of enemy kill assists provided by this Player in this Game.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists received by this Player in this Game.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this Player in this Game.
headshots
Int!
Number of enemy headshots for this Player.
teamkills
Int!
Number of teamkills for this Player in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this Player in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this Player in this Game.A teamkill is the occurrence of killing an allied member
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists received by this Player in this Game.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of teamkills, which were executed with a specific weapon by this Player in this Game.A teamkill is the occurrence of killing an allied member.
teamHeadshots
Int!
Number of allied headshots for this Player.(available until version "1.9")
selfkills
Int!
Number of selfkills (suicides) for this Player in this Game.
deaths
Int!
Number of deaths for this Player in this Game.
firstKill
Boolean!
Indication of player acquiring first kill.
structuresDestroyed
Int!
Number of structures destroyed by this Player in this Game.
structuresCaptured
Int!
Number of structures captured by this Player in this Game.
inventory
PlayerInventory!
The Player's inventory
objectives
[Objective!]!
A list of objectives that were fulfilled by this Player in this Game.
position
Coordinates
Player Coordinates on the map
multikills
[Multikill!]!
A list of multi kills that happened by this Player in this Game.
unitKills
[UnitKill!]!
A list of unit kills that happened by this Player in this Game.
abilities
[DefaultAbility!]!
A list of abilities that are owned by this Player in this Game.
statusEffects
[StatusEffect!]!
A list of status effects affecting the player
alive
Boolean!
Indicates whether the player is alive.
respawnClock
ClockState!
Respawn clock state to indicate when this player respawns.(available from version "3.45")
experiencePoints
Int!
The amount of experience points gathered by this player.(available from version "3.46")
currentHealth
Int!
The current health of the player.(available from version "3.47")
maxHealth
Int!
The max amount of health of the player.(available from version "3.47")
GamePlayerStatePubg
PUBG data points for a Player, aggregated for a Game. (available from version "1.1")
Field
Type
Description
id
ID!
GRID ID of this Player.
name
String!
Name of this Player.
character
Character
In-game character (ie champion, class) of the Player in this Game
roles
[PlayerRole!]!
Roles this Player has in this Game.(available from version "3.43")
participationStatus
ParticipationStatus!
Indicates the participation status of a player for this Game
money
Int!
Amount of money this Player is holding in cash.Depending on the Title this may represent in-game gold, credits, gems etc.
loadoutValue
Int!
Total value (worth in money) of this Player’s loadout.
netWorth
Int!
Sum of money and inventoryValue for this Player.
kills
Int!
Number of enemy kills for this Player in this Game.
killAssistsReceived
Int!
Number of enemy kill assists received by this Player in this Game.
killAssistsGiven
Int!
Number of enemy kill assists provided by this Player in this Game.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists received by this Player in this Game.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this Player in this Game.
teamkills
Int!
Number of teamkills for this Player in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this Player in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this Player in this Game.A teamkill is the occurrence of killing an allied member
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists received by this Player in this Game.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of teamkills, which were executed with a specific weapon by this Player in this Game.A teamkill is the occurrence of killing an allied member.
selfkills
Int!
Number of selfkills (suicides) for this Player in this Game.
deaths
Int!
Number of deaths for this Player in this Game.
firstKill
Boolean!
Indication of player acquiring first kill.(available from version "3.10")
structuresDestroyed
Int!
Number of structures destroyed by this Player in this Game.
structuresCaptured
Int!
Number of structures captured by this Player in this Game.
inventory
PlayerInventory!
The Player's inventory
objectives
[Objective!]!
A list of objectives that were fulfilled by this Player in this Game.
position
Coordinates
Player Coordinates on the map
multikills
[Multikill!]!
A list of multi kills that happened by this Player in this Game.(available from version "1.7")
unitKills
[UnitKill!]!
A list of unit kills that happened by this Player in this Game.(available from version "3.1")
abilities
[DefaultAbility!]!
A list of abilities that are owned by this Player in this Game.(available from version "1.8")
statusEffects
[StatusEffect!]!
A list of status effects affecting the player(available from version "3.31")
alive
Boolean!
Indicates whether the player is alive.
currentHealth
Int!
The current health of the player.
maxHealth
Int!
The max amount of health of the player.
headshots
Int!
Number of enemy headshots for this Player.
teamHeadshots
Int!
Number of allied headshots for this Player.
GamePlayerStateR6
R6 data points for a Player, aggregated for a Game
Field
Type
Description
id
ID!
GRID ID of this Player.
name
String!
Name of this Player.
character
Character
In-game character (ie champion, class) of the Player in this Game
roles
[PlayerRole!]!
Roles this Player has in this Game.(available from version "3.43")
participationStatus
ParticipationStatus!
Indicates the participation status of a player for this Game
money
Int!
Amount of money this Player is holding in cash.Depending on the Title this may represent in-game gold, credits, gems etc.
loadoutValue
Int!
Total value (worth in money) of this Player’s loadout.
netWorth
Int!
Sum of money and inventoryValue for this Player.
kills
Int!
Number of enemy kills for this Player in this Game.
killAssistsReceived
Int!
Number of enemy kill assists received by this Player in this Game.
killAssistsGiven
Int!
Number of enemy kill assists provided by this Player in this Game.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists received by this Player in this Game.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this Player in this Game.
headshots
Int!
Number of enemy headshots for this Player.(available until version "1.9")
teamkills
Int!
Number of teamkills for this Player in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this Player in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this Player in this Game.A teamkill is the occurrence of killing an allied member
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists received by this Player in this Game.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of teamkills, which were executed with a specific weapon by this Player in this Game.A teamkill is the occurrence of killing an allied member.
teamHeadshots
Int!
Number of allied headshots for this Player.(available until version "1.9")
selfkills
Int!
Number of selfkills (suicides) for this Player in this Game.
deaths
Int!
Number of deaths for this Player in this Game.
firstKill
Boolean!
Indication of player acquiring first kill.(available from version "3.10")
structuresDestroyed
Int!
Number of structures destroyed by this Player in this Game.
structuresCaptured
Int!
Number of structures captured by this Player in this Game.
inventory
PlayerInventory!
The Player's inventory
objectives
[Objective!]!
A list of objectives that were fulfilled by this Player in this Game.
position
Coordinates
Player Coordinates on the map
multikills
[Multikill!]!
A list of multi kills that happened by this Player in this Game.(available from version "1.7")
unitKills
[UnitKill!]!
A list of unit kills that happened by this Player in this Game.(available from version "3.1")
abilities
[DefaultAbility!]!
A list of abilities that are owned by this Player in this Game.(available from version "1.8")
statusEffects
[StatusEffect!]!
A list of status effects affecting the player(available from version "3.31")
alive
Boolean!
Indicates whether the player is alive.
currentHealth
Int!
The current health of the player.
damageDealt
Int!
The amount of damage dealt.
damageTaken
Int!
The amount of total damage taken.
maxHealth
Int!
The max amount of health of the player.
selfdamageDealt
Int!
The amount of damage dealt to self.
selfdamageTaken
Int!
The amount of damage taken from self.
teamdamageDealt
Int!
The amount of damage dealt to team.
teamdamageTaken
Int!
The amount of damage taken from team.
healingDealt
Int!
The amount of healing dealt by team.(available from version "3.5")
healingReceived
Int!
The amount of healing received by team.(available from version "3.5")
knockdownsGiven
Int!
Number of times of knocking down an enemy for this player.(available from version "3.9")
knockdownsReceived
Int!
Number of times of being knocked down for this player.(available from version "3.9")
damageDealtSources
[DamageDealtSource!]!
A list of damage sources.(available from version "3.17")
damageDealtTargets
[DamageDealtTarget!]!
A list of damage targets.(available from version "3.17")
GamePlayerStateValorant
Valorant data points for a Player, aggregated for a Game. (available from version "1.1")
Field
Type
Description
id
ID!
GRID ID of this Player.
name
String!
Name of this Player.
character
Character
In-game character (ie champion, class) of the Player in this Game
roles
[PlayerRole!]!
Roles this Player has in this Game.(available from version "3.43")
participationStatus
ParticipationStatus!
Indicates the participation status of a player for this Game
money
Int!
Amount of money this Player is holding in cash.Depending on the Title this may represent in-game gold, credits, gems etc.
loadoutValue
Int!
Total value (worth in money) of this Player’s loadout.
netWorth
Int!
Sum of money and inventoryValue for this Player.
kills
Int!
Number of enemy kills for this Player in this Game.
killAssistsReceived
Int!
Number of enemy kill assists received by this Player in this Game.
killAssistsGiven
Int!
Number of enemy kill assists provided by this Player in this Game.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists received by this Player in this Game.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this Player in this Game.
teamkills
Int!
Number of teamkills for this Player in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this Player in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this Player in this Game.A teamkill is the occurrence of killing an allied member
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists received by this Player in this Game.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of teamkills, which were executed with a specific weapon by this Player in this Game.A teamkill is the occurrence of killing an allied member.
selfkills
Int!
Number of selfkills (suicides) for this Player in this Game.
deaths
Int!
Number of deaths for this Player in this Game.
firstKill
Boolean!
Indication of player acquiring first kill.(available from version "3.10")
structuresDestroyed
Int!
Number of structures destroyed by this Player in this Game.
structuresCaptured
Int!
Number of structures captured by this Player in this Game.
inventory
PlayerInventory!
The Player's inventory
objectives
[Objective!]!
A list of objectives that were fulfilled by this Player in this Game.
position
Coordinates
Player Coordinates on the map
multikills
[Multikill!]!
A list of multi kills that happened by this Player in this Game.(available from version "1.7")
unitKills
[UnitKill!]!
A list of unit kills that happened by this Player in this Game.(available from version "3.1")
abilities
[AbilityValorant!]!
A list of abilities that are owned by this Player in this Game.(available from version "1.8")
statusEffects
[StatusEffect!]!
A list of status effects affecting the player(available from version "3.31")
alive
Boolean!
Indicates whether the player is alive.(available from version "1.9")
currentArmor
Int!
The amount of current armor.(available from version "1.9")
currentHealth
Int!
The current health of the player.(available from version "1.9")
maxHealth
Int!
The max amount of health of the player.(available from version "1.9")
headshots
Int!
Number of enemy headshots for this Player.
teamHeadshots
Int!
Number of team headshots for this Player.
ultimatePoints
Int!
Number of ultimate points of this Player.(available from version "3.20")
damageDealt
Int!
The amount of damage dealt.(available from version "3.29")
damageTaken
Int!
The amount of total damage taken.(available from version "3.29")
selfdamageDealt
Int!
The amount of damage dealt to self.(available from version "3.29")
selfdamageTaken
Int!
The amount of damage taken from self.(available from version "3.29")
teamdamageDealt
Int!
The amount of damage dealt to team.(available from version "3.29")
teamdamageTaken
Int!
The amount of damage taken from team.(available from version "3.29")
damageDealtSources
[DamageDealtSource!]!
A list of damage sources.(available from version "3.29")
damageDealtTargets
[DamageDealtTarget!]!
A list of damage targets.(available from version "3.29")
GameState
Data points for a Game.
Field
Type
Description
id
ID!
GRID ID of this Game.
sequenceNumber
Int!
SequenceNumber of the Game state update.
map
MapState!
Information on the map of this Game.
titleVersion
TitleVersion
Indicates the game version(available from version "3.12")
type
GameType
Indicates the game type(available from version "3.21")
started
Boolean!
Indicates whether this Game has started.
finished
Boolean!
Indicates whether this Game has finished.
forfeited
Boolean!
Indicates whether this Game was forfeited.(available from version "3.48")
paused
Boolean!
Indicates whether this Game is paused.
startedAt
DateTime
DateTime value when this Game was started.(available from version "3.7")
clock
ClockState
Clock state to indicate time of the game(available from version "1.3")
structures
[StructureState!]!
A list of StructureState, containing information about the structures that are available in this Game.
nonPlayerCharacters
[NonPlayerCharacterState!]!
A list of NonPlayerCharacterState, containing information about the state of NPCs in this Game.
teams
[GameTeamState!]!
A list of GameTeamState, containing information on the teams participating in this Game.
draftActions
[DraftAction!]!
A list of DraftAction, containing information about draft actions in this Game.
externalLinks
[ExternalLink!]!
A list of ExternalLink, containing information about links to externally provided IDs.(available from version "1.2")
segments
[SegmentState!]!
A list of SegmentStates, containing information of the segments in this Game.(available from version "1.4")
duration
Duration!
Duration of the Game.(available from version "3.15")
GameTeamStateCs2
CS2 data points for a Team, aggregated for a Game.
Field
Type
Description
id
ID!
GRID ID of this Team.
name
String!
Name of this Team.
side
String!
Side that this Team has taken in this Game ie Red or Blue, Terrorists or Counter-Terrorists.
won
Boolean!
Indicates whether this Team has won this Game.
score
Int!
Score for this Team in this Game. This is used in the Score After format.
money
Int!
Amount of money this Team is holding in cash.Depending on the Title this may represent in-game gold, credits, gems etc.
loadoutValue
Int!
Total value (worth in money) of all the loadouts owned by Players that are members of this Team.
netWorth
Int!
Sum of money and inventoryValue for this Team.
kills
Int!
Number of enemy kills for this Team in this Game.
killAssistsReceived
Int!
Number of enemy kill assists received by this Team in this Game.
killAssistsGiven
Int!
Number of enemy kill assists provided by this Team in this Game.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists from a Player, received by this Team in this Game.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this Team in this Game.
teamkills
Int!
Number of teamkills for this Team in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this Team in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this Team in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists from a Player, received by this Team in this Game.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of teamkills, which were executed with a specific weapon by this Team in this Game.A teamkill is the occurrence of killing an allied member.
selfkills
Int!
Number of selfkills (suicides) for this Team in this Game.
deaths
Int!
Number of deaths for this Team in this Game.
firstKill
Boolean!
Indication of team acquiring first kill.(available from version "3.10")
structuresDestroyed
Int!
Number of structures destroyed by this Team in this Game.
structuresCaptured
Int!
Number of structures captured by this Team in this Game.
objectives
[Objective!]!
A list of objectives that were fulfilled by this Team in this Game.
unitKills
[UnitKill!]!
A list of unit kills that happened by this Team in this Game.(available from version "3.1")
players
[GamePlayerState!]!
A list of GamePlayerState, containing information about the Players that are members of this Team.
multikills
[Multikill!]!
A list of multi kills that happened by this Team in this Game.(available from version "1.7")
damageDealt
Int!
The amount of damage dealt.
damageTaken
Int!
The amount of total damage taken.
selfdamageDealt
Int!
The amount of damage dealt to self.
selfdamageTaken
Int!
The amount of damage taken from self.
teamdamageDealt
Int!
The amount of damage dealt to team.
teamdamageTaken
Int!
The amount of damage taken from team.
headshots
Int!
Number of enemy headshots for this Player.
teamHeadshots
Int!
Number of allied headshots for this Player.
damageDealtSources
[DamageDealtSource!]!
A list of damage sources.(available from version "3.17")
damageDealtTargets
[DamageDealtTarget!]!
A list of damage targets.(available from version "3.17")
GameTeamStateCsgo
CSGO data points for a Team, aggregated for a Game. (available from version "1.1")
Field
Type
Description
id
ID!
GRID ID of this Team.
name
String!
Name of this Team.
side
String!
Side that this Team has taken in this Game ie Red or Blue, Terrorists or Counter-Terrorists.
won
Boolean!
Indicates whether this Team has won this Game.
score
Int!
Score for this Team in this Game. This is used in the Score After format.
money
Int!
Amount of money this Team is holding in cash.Depending on the Title this may represent in-game gold, credits, gems etc.
loadoutValue
Int!
Total value (worth in money) of all the loadouts owned by Players that are members of this Team.
netWorth
Int!
Sum of money and inventoryValue for this Team.
kills
Int!
Number of enemy kills for this Team in this Game.
killAssistsReceived
Int!
Number of enemy kill assists received by this Team in this Game.
killAssistsGiven
Int!
Number of enemy kill assists provided by this Team in this Game.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists from a Player, received by this Team in this Game.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this Team in this Game.
teamkills
Int!
Number of teamkills for this Team in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this Team in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this Team in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists from a Player, received by this Team in this Game.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of teamkills, which were executed with a specific weapon by this Team in this Game.A teamkill is the occurrence of killing an allied member.
selfkills
Int!
Number of selfkills (suicides) for this Team in this Game.
deaths
Int!
Number of deaths for this Team in this Game.
firstKill
Boolean!
Indication of team acquiring first kill.(available from version "3.10")
structuresDestroyed
Int!
Number of structures destroyed by this Team in this Game.
structuresCaptured
Int!
Number of structures captured by this Team in this Game.
objectives
[Objective!]!
A list of objectives that were fulfilled by this Team in this Game.
unitKills
[UnitKill!]!
A list of unit kills that happened by this Team in this Game.(available from version "3.1")
players
[GamePlayerState!]!
A list of GamePlayerState, containing information about the Players that are members of this Team.
multikills
[Multikill!]!
A list of multi kills that happened by this Team in this Game.(available from version "1.7")
damageDealt
Int!
The amount of damage dealt.
damageTaken
Int!
The amount of total damage taken.
selfdamageDealt
Int!
The amount of damage dealt to self.
selfdamageTaken
Int!
The amount of damage taken from self.
teamdamageDealt
Int!
The amount of damage dealt to team.
teamdamageTaken
Int!
The amount of damage taken from team.
headshots
Int!
Number of enemy headshots for this Player.
teamHeadshots
Int!
Number of allied headshots for this Player.
damageDealtSources
[DamageDealtSource!]!
A list of damage sources.(available from version "3.17")
damageDealtTargets
[DamageDealtTarget!]!
A list of damage targets.(available from version "3.17")
GameTeamStateDefault
Field
Type
Description
id
ID!
GRID ID of this Team.
name
String!
Name of this Team.
side
String!
Side that this Team has taken in this Game ie Red or Blue, Terrorists or Counter-Terrorists.
won
Boolean!
Indicates whether this Team has won this Game.
score
Int!
Score for this Team in this Game. This is used in the Score After format.
money
Int!
Amount of money this Team is holding in cash.Depending on the Title this may represent in-game gold, credits, gems etc.
loadoutValue
Int!
Total value (worth in money) of all the loadouts owned by Players that are members of this Team.
netWorth
Int!
Sum of money and inventoryValue for this Team.
kills
Int!
Number of enemy kills for this Team in this Game.
killAssistsReceived
Int!
Number of enemy kill assists received by this Team in this Game.
killAssistsGiven
Int!
Number of enemy kill assists provided by this Team in this Game.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists from a Player, received by this Team in this Game.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this Team in this Game.
headshots
Int!
Number of enemy headshots for this Player.(available until version "1.9")
teamkills
Int!
Number of teamkills for this Team in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this Team in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this Team in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists from a Player, received by this Team in this Game.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of teamkills, which were executed with a specific weapon by this Team in this Game.A teamkill is the occurrence of killing an allied member.
teamHeadshots
Int!
Number of allied headshots for this Player.(available until version "1.9")
selfkills
Int!
Number of selfkills (suicides) for this Team in this Game.
deaths
Int!
Number of deaths for this Team in this Game.
firstKill
Boolean!
Indication of team acquiring first kill.(available from version "3.10")
structuresDestroyed
Int!
Number of structures destroyed by this Team in this Game.
structuresCaptured
Int!
Number of structures captured by this Team in this Game.
objectives
[Objective!]!
A list of objectives that were fulfilled by this Team in this Game.
unitKills
[UnitKill!]!
A list of unit kills that happened by this Team in this Game.(available from version "3.1")
players
[GamePlayerState!]!
A list of GamePlayerState, containing information about the Players that are members of this Team.
multikills
[Multikill!]!
A list of multi kills that happened by this Team in this Game.(available from version "1.7")
GameTeamStateDota
Dota data points for a Team, aggregated for a Game. (available from version "1.1")
Field
Type
Description
id
ID!
GRID ID of this Team.
name
String!
Name of this Team.
side
String!
Side that this Team has taken in this Game ie Red or Blue, Terrorists or Counter-Terrorists.
won
Boolean!
Indicates whether this Team has won this Game.
score
Int!
Score for this Team in this Game. This is used in the Score After format.
money
Int!
Amount of money this Team is holding in cash.Depending on the Title this may represent in-game gold, credits, gems etc.
loadoutValue
Int!
Total value (worth in money) of all the loadouts owned by Players that are members of this Team.
netWorth
Int!
Sum of money and inventoryValue for this Team.
kills
Int!
Number of enemy kills for this Team in this Game.
killAssistsReceived
Int!
Number of enemy kill assists received by this Team in this Game.
killAssistsGiven
Int!
Number of enemy kill assists provided by this Team in this Game.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists from a Player, received by this Team in this Game.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this Team in this Game.
headshots
Int!
Number of enemy headshots for this Player.(available until version "1.9")
teamkills
Int!
Number of teamkills for this Team in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this Team in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this Team in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists from a Player, received by this Team in this Game.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of teamkills, which were executed with a specific weapon by this Team in this Game.A teamkill is the occurrence of killing an allied member.
teamHeadshots
Int!
Number of allied headshots for this Player.(available until version "1.9")
selfkills
Int!
Number of selfkills (suicides) for this Team in this Game.
deaths
Int!
Number of deaths for this Team in this Game.
firstKill
Boolean!
Indication of team acquiring first kill.(available from version "3.10")
structuresDestroyed
Int!
Number of structures destroyed by this Team in this Game.
structuresCaptured
Int!
Number of structures captured by this Team in this Game.
objectives
[Objective!]!
A list of objectives that were fulfilled by this Team in this Game.
unitKills
[UnitKill!]!
A list of unit kills that happened by this Team in this Game.(available from version "3.1")
players
[GamePlayerState!]!
A list of GamePlayerState, containing information about the Players that are members of this Team.
multikills
[Multikill!]!
A list of multi kills that happened by this Team in this Game.(available from version "1.7")
experiencePoints
Int!
The amount of experience points gathered by this team.
GameTeamStateLol
LoL data points for a Team, aggregated for a Game. (available from version "1.1")
Field
Type
Description
id
ID!
GRID ID of this Team.
name
String!
Name of this Team.
side
String!
Side that this Team has taken in this Game ie Red or Blue, Terrorists or Counter-Terrorists.
won
Boolean!
Indicates whether this Team has won this Game.
score
Int!
Score for this Team in this Game. This is used in the Score After format.
money
Int!
Amount of money this Team is holding in cash.Depending on the Title this may represent in-game gold, credits, gems etc.
loadoutValue
Int!
Total value (worth in money) of all the loadouts owned by Players that are members of this Team.
netWorth
Int!
Sum of money and inventoryValue for this Team.
kills
Int!
Number of enemy kills for this Team in this Game.
killAssistsReceived
Int!
Number of enemy kill assists received by this Team in this Game.
killAssistsGiven
Int!
Number of enemy kill assists provided by this Team in this Game.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists from a Player, received by this Team in this Game.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this Team in this Game.
headshots
Int!
Number of enemy headshots for this Player.(available until version "1.9")
teamkills
Int!
Number of teamkills for this Team in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this Team in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this Team in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists from a Player, received by this Team in this Game.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of teamkills, which were executed with a specific weapon by this Team in this Game.A teamkill is the occurrence of killing an allied member.
teamHeadshots
Int!
Number of allied headshots for this Player.(available until version "1.9")
selfkills
Int!
Number of selfkills (suicides) for this Team in this Game.
deaths
Int!
Number of deaths for this Team in this Game.
firstKill
Boolean!
Indication of team acquiring first kill.(available from version "3.10")
structuresDestroyed
Int!
Number of structures destroyed by this Team in this Game.
structuresCaptured
Int!
Number of structures captured by this Team in this Game.
objectives
[Objective!]!
A list of objectives that were fulfilled by this Team in this Game.
unitKills
[UnitKill!]!
A list of unit kills that happened by this Team in this Game.(available from version "3.1")
players
[GamePlayerState!]!
A list of GamePlayerState, containing information about the Players that are members of this Team.
multikills
[Multikill!]!
A list of multi kills that happened by this Team in this Game.(available from version "1.7")
baronPowerPlays
[LolBaronPowerPlay!]!
A list of the Baron power plays that were completed by this team.(available from version "3.38")
damageDealt
Int!
The amount of damage dealt.(available from version "3.23")
damagePerMinute
Float!
The amount of damage dealt per minute.(available from version "3.24")
damagePerMoney
Float!
The amount of damage dealt per money earned.(available from version "3.26")
damageTaken
Int!
The amount of damage taken.(available from version "3.23")
experiencePoints
Int!
The amount of experience points gathered by this team.
forwardPercentage
Float!
The average percentage of game time each player of this team spent on the opponent's side of the map.(available from version "3.42")
kdaRatio
Float!
The ratio of kills and assists given vs deaths.(available from version "3.27")
killsAndAssists
Float!
The sum of kills and assists given.(available from version "3.34")
majorMoneyDeficit
Float!
The percentage of game time this team owned 48.5% or less of the overall game's gold up to the 40th minute.(available from version "3.41")
majorMoneyLead
Float!
The percentage of game time this team owned 51.5% or more of the overall game's gold up to the 40th minute.(available from version "3.40")
moneyDifference
Int!
The total money earned difference to the opponent team.(available from version "3.28")
moneyPerMinute
Float!
The amount of money earned per minute.(available from version "3.36")
totalMoneyEarned
Int!
The total amount of money that was earned by this team.(available from version "3.2")
visionScore
Float!
Indicates how much vision this team has influenced in the game, including the vision it granted and denied.(available from version "3.30")
visionScorePerMinute
Float!
The vision score gained per minute.(available from version "3.33")
GameTeamStatePubg
PUBG data points for a Team, aggregated for a Game. (available from version "1.1")
Field
Type
Description
id
ID!
GRID ID of this Team.
name
String!
Name of this Team.
side
String!
Side that this Team has taken in this Game ie Red or Blue, Terrorists or Counter-Terrorists.
won
Boolean!
Indicates whether this Team has won this Game.
score
Int!
Score for this Team in this Game. This is used in the Score After format.
money
Int!
Amount of money this Team is holding in cash.Depending on the Title this may represent in-game gold, credits, gems etc.
loadoutValue
Int!
Total value (worth in money) of all the loadouts owned by Players that are members of this Team.
netWorth
Int!
Sum of money and inventoryValue for this Team.
kills
Int!
Number of enemy kills for this Team in this Game.
killAssistsReceived
Int!
Number of enemy kill assists received by this Team in this Game.
killAssistsGiven
Int!
Number of enemy kill assists provided by this Team in this Game.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists from a Player, received by this Team in this Game.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this Team in this Game.
teamkills
Int!
Number of teamkills for this Team in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this Team in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this Team in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists from a Player, received by this Team in this Game.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of teamkills, which were executed with a specific weapon by this Team in this Game.A teamkill is the occurrence of killing an allied member.
selfkills
Int!
Number of selfkills (suicides) for this Team in this Game.
deaths
Int!
Number of deaths for this Team in this Game.
firstKill
Boolean!
Indication of team acquiring first kill.(available from version "3.10")
structuresDestroyed
Int!
Number of structures destroyed by this Team in this Game.
structuresCaptured
Int!
Number of structures captured by this Team in this Game.
objectives
[Objective!]!
A list of objectives that were fulfilled by this Team in this Game.
unitKills
[UnitKill!]!
A list of unit kills that happened by this Team in this Game.(available from version "3.1")
players
[GamePlayerState!]!
A list of GamePlayerState, containing information about the Players that are members of this Team.
multikills
[Multikill!]!
A list of multi kills that happened by this Team in this Game.(available from version "1.7")
headshots
Int!
Number of enemy headshots for this Player.
teamHeadshots
Int!
Number of allied headshots for this Player.
GameTeamStateR6
Field
Type
Description
id
ID!
GRID ID of this Team.
name
String!
Name of this Team.
side
String!
Side that this Team has taken in this Game ie Red or Blue, Terrorists or Counter-Terrorists.
won
Boolean!
Indicates whether this Team has won this Game.
score
Int!
Score for this Team in this Game. This is used in the Score After format.
money
Int!
Amount of money this Team is holding in cash.Depending on the Title this may represent in-game gold, credits, gems etc.
loadoutValue
Int!
Total value (worth in money) of all the loadouts owned by Players that are members of this Team.
netWorth
Int!
Sum of money and inventoryValue for this Team.
kills
Int!
Number of enemy kills for this Team in this Game.
killAssistsReceived
Int!
Number of enemy kill assists received by this Team in this Game.
killAssistsGiven
Int!
Number of enemy kill assists provided by this Team in this Game.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists from a Player, received by this Team in this Game.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this Team in this Game.
headshots
Int!
Number of enemy headshots for this Player.(available until version "1.9")
teamkills
Int!
Number of teamkills for this Team in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this Team in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this Team in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists from a Player, received by this Team in this Game.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of teamkills, which were executed with a specific weapon by this Team in this Game.A teamkill is the occurrence of killing an allied member.
teamHeadshots
Int!
Number of allied headshots for this Player.(available until version "1.9")
selfkills
Int!
Number of selfkills (suicides) for this Team in this Game.
deaths
Int!
Number of deaths for this Team in this Game.
firstKill
Boolean!
Indication of team acquiring first kill.(available from version "3.10")
structuresDestroyed
Int!
Number of structures destroyed by this Team in this Game.
structuresCaptured
Int!
Number of structures captured by this Team in this Game.
objectives
[Objective!]!
A list of objectives that were fulfilled by this Team in this Game.
unitKills
[UnitKill!]!
A list of unit kills that happened by this Team in this Game.(available from version "3.1")
players
[GamePlayerState!]!
A list of GamePlayerState, containing information about the Players that are members of this Team.
multikills
[Multikill!]!
A list of multi kills that happened by this Team in this Game.(available from version "1.7")
damageDealt
Int!
The amount of damage dealt.
damageTaken
Int!
The amount of total damage taken.
selfdamageDealt
Int!
The amount of damage dealt to self.
selfdamageTaken
Int!
The amount of damage taken from self.
teamdamageDealt
Int!
The amount of damage dealt to team.
teamdamageTaken
Int!
The amount of damage taken from team.
healingDealt
Int!
The amount of healing dealt by team.(available from version "3.5")
healingReceived
Int!
The amount of healing received by team.(available from version "3.5")
knockdownsGiven
Int!
Number of times of knocking down an enemy for this team.(available from version "3.9")
knockdownsReceived
Int!
Number of times of being knocked down for this team.(available from version "3.9")
damageDealtSources
[DamageDealtSource!]!
A list of damage sources.(available from version "3.17")
damageDealtTargets
[DamageDealtTarget!]!
A list of damage targets.(available from version "3.17")
GameTeamStateValorant
Valorant data points for a Team, aggregated for a Game. (available from version "1.1")
Field
Type
Description
id
ID!
GRID ID of this Team.
name
String!
Name of this Team.
side
String!
Side that this Team has taken in this Game ie Red or Blue, Terrorists or Counter-Terrorists.
won
Boolean!
Indicates whether this Team has won this Game.
score
Int!
Score for this Team in this Game. This is used in the Score After format.
money
Int!
Amount of money this Team is holding in cash.Depending on the Title this may represent in-game gold, credits, gems etc.
loadoutValue
Int!
Total value (worth in money) of all the loadouts owned by Players that are members of this Team.
netWorth
Int!
Sum of money and inventoryValue for this Team.
kills
Int!
Number of enemy kills for this Team in this Game.
killAssistsReceived
Int!
Number of enemy kill assists received by this Team in this Game.
killAssistsGiven
Int!
Number of enemy kill assists provided by this Team in this Game.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists from a Player, received by this Team in this Game.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this Team in this Game.
teamkills
Int!
Number of teamkills for this Team in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this Team in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this Team in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists from a Player, received by this Team in this Game.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of teamkills, which were executed with a specific weapon by this Team in this Game.A teamkill is the occurrence of killing an allied member.
selfkills
Int!
Number of selfkills (suicides) for this Team in this Game.
deaths
Int!
Number of deaths for this Team in this Game.
firstKill
Boolean!
Indication of team acquiring first kill.(available from version "3.10")
structuresDestroyed
Int!
Number of structures destroyed by this Team in this Game.
structuresCaptured
Int!
Number of structures captured by this Team in this Game.
objectives
[Objective!]!
A list of objectives that were fulfilled by this Team in this Game.
unitKills
[UnitKill!]!
A list of unit kills that happened by this Team in this Game.(available from version "3.1")
players
[GamePlayerState!]!
A list of GamePlayerState, containing information about the Players that are members of this Team.
multikills
[Multikill!]!
A list of multi kills that happened by this Team in this Game.(available from version "1.7")
headshots
Int!
Number of enemy headshots for this Player.
teamHeadshots
Int!
Number of allied headshots for this Player.
damageDealt
Int!
The amount of damage dealt.(available from version "3.29")
damageTaken
Int!
The amount of total damage taken.(available from version "3.29")
selfdamageDealt
Int!
The amount of damage dealt to self.(available from version "3.29")
selfdamageTaken
Int!
The amount of damage taken from self.(available from version "3.29")
teamdamageDealt
Int!
The amount of damage dealt to team.(available from version "3.29")
teamdamageTaken
Int!
The amount of damage taken from team.(available from version "3.29")
damageDealtSources
[DamageDealtSource!]!
A list of damage sources.(available from version "3.29")
damageDealtTargets
[DamageDealtTarget!]!
A list of damage targets.(available from version "3.29")
ItemStack
An item stack that is part of a player's inventory.
Field
Type
Description
id
ID!
GRID ID for this item.
name
String!
The name of this item.
quantity
Int!
The amount of items in this stack (includes equipped and stashed items).
equipped
Int!
The amount of equipped items.
stashed
Int!
The amount of stashed items.
KillAssistFromPlayer
Kill assist information.
Field
Type
Description
id
ID!
GRID ID for this assist.
playerId
ID!
GRID Player ID assisting.
killAssistsReceived
Int!
Number of kill assists received from the assisting Player.
LolBaronPowerPlay
(available from version "3.38")
Field
Type
Description
id
ID!
GRID ID for this Baron power play
value
Int!
Value of the Baron power play after it has ended
MapState
Map information
Field
Type
Description
id
String!
GRID Map ID.(available from version "3.13")
name
String!
Map name
bounds
Bounds
Map Bounds information
Multikill
A multikill caused by a player or a team. (available from version "1.7")
Field
Type
Description
id
ID!
GRID ID for this multikill
numberOfKills
Int!
The type of multikill
count
Int!
Amount of times a specific multikill has happened
NonPlayerCharacterState
Data points for Non Playing Characters (NPCs).
Field
Type
Description
id
ID!
GRID ID of the NPC.
type
String!
Type of the NPC
side
String!
Side that controls the NPC
respawnClock
ClockState
Respawn clock state to indicate when the NPC respawns.(available from version "1.3")
position
Coordinates
NPC Coordinates on the map.
alive
Boolean!
Indicates whether the NPC is alive.
Objective
An objective that shall be finished.
Field
Type
Description
id
ID!
GRID ID for this objective.
type
String!
The objective type.
completedFirst
Boolean!
Mark that the objective was completed for the first time on the current level.(available from version "3.11")
completionCount
Int!
Amount of times this objective was completed.
PlayerInventory
The inventory of a Player.
Field
Type
Description
items
[ItemStack!]!
The items that are contained in the Player's inventory.
PlayerRole
A role a player has in the current match. The IDs can be looked up in the content catalogs of the corresponding game title. (available from version "3.43")
Field
Type
Description
id
ID!
GRID Player Role ID
SegmentPlayerStateCs2
CS2 data points for a Player, aggregated for a Segment.
Field
Type
Description
id
ID!
GRID ID of this Player.
name
String!
Name of this Player.
participationStatus
ParticipationStatus!
Indicates the participation status of a player for this Segment.(available from version "3.8")
kills
Int!
Number of enemy kills for this Player in this Segment.
killAssistsReceived
Int!
Number of enemy kill assists received by this Player in this Segment.
killAssistsGiven
Int!
Number of enemy kill assists provided by this Player in this Segment.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists received by this Player in this Segment.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this Player in this Segment.
teamkills
Int!
Number of teamkills for this Player in this Segment.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this Player in this Segment.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this Player in this Segment.A teamkill is the occurrence of killing an allied member
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists received by this Player in this Segment.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of team kills, which were executed with a specific weapon by this Player in this Segment.
selfkills
Int!
Number of selfkills (suicides) for this Player in this Segment.
deaths
Int!
Number of deaths for this Player in this Segment.
firstKill
Boolean!
Indication of player acquiring first kill.(available from version "3.10")
objectives
[Objective!]!
A list of objectives that were fulfilled by this Player in this Segment.
headshots
Int!
Number of enemy headshots for this Player.
teamHeadshots
Int!
Number of allied headshots for this Player.
alive
Boolean!
Indicates whether the player is alive.
currentArmor
Int!
The amount of current armor.
currentHealth
Int!
The current health of the player.
damageDealt
Int!
The amount of damage dealt.
damageTaken
Int!
The amount of total damage taken.
maxHealth
Int!
The max amount of health of the player.
selfdamageDealt
Int!
The amount of damage dealt to self.
selfdamageTaken
Int!
The amount of damage taken from self.
teamdamageDealt
Int!
The amount of damage dealt to team.
teamdamageTaken
Int!
The amount of damage taken from team.
externalLinks
[ExternalLink!]!
A list of ExternalLink, containing information about links to externally provided IDs.
damageDealtSources
[DamageDealtSource!]!
A list of damage sources.(available from version "3.17")
damageDealtTargets
[DamageDealtTarget!]!
A list of damage targets.(available from version "3.17")
SegmentPlayerStateCsgo
CSGO data points for a Player, aggregated for a Segment. (available from version "1.4")
Field
Type
Description
id
ID!
GRID ID of this Player.
name
String!
Name of this Player.
participationStatus
ParticipationStatus!
Indicates the participation status of a player for this Segment.(available from version "3.8")
kills
Int!
Number of enemy kills for this Player in this Segment.
killAssistsReceived
Int!
Number of enemy kill assists received by this Player in this Segment.
killAssistsGiven
Int!
Number of enemy kill assists provided by this Player in this Segment.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists received by this Player in this Segment.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this Player in this Segment.
teamkills
Int!
Number of teamkills for this Player in this Segment.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this Player in this Segment.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this Player in this Segment.A teamkill is the occurrence of killing an allied member
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists received by this Player in this Segment.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of team kills, which were executed with a specific weapon by this Player in this Segment.
selfkills
Int!
Number of selfkills (suicides) for this Player in this Segment.
deaths
Int!
Number of deaths for this Player in this Segment.
firstKill
Boolean!
Indication of player acquiring first kill.(available from version "3.10")
objectives
[Objective!]!
A list of objectives that were fulfilled by this Player in this Segment.
headshots
Int!
Number of enemy headshots for this Player.
teamHeadshots
Int!
Number of allied headshots for this Player.
alive
Boolean!
Indicates whether the player is alive.
currentArmor
Int!
The amount of current armor.
currentHealth
Int!
The current health of the player.
damageDealt
Int!
The amount of damage dealt.
damageTaken
Int!
The amount of total damage taken.
maxHealth
Int!
The max amount of health of the player.
selfdamageDealt
Int!
The amount of damage dealt to self.
selfdamageTaken
Int!
The amount of damage taken from self.
teamdamageDealt
Int!
The amount of damage dealt to team.
teamdamageTaken
Int!
The amount of damage taken from team.
externalLinks
[ExternalLink!]!
A list of ExternalLink, containing information about links to externally provided IDs.(available from version "2.2")
damageDealtSources
[DamageDealtSource!]!
A list of damage sources.(available from version "3.17")
damageDealtTargets
[DamageDealtTarget!]!
A list of damage targets.(available from version "3.17")
SegmentPlayerStateDefault
Default data points for a Player, aggregated for a Segment (available from version "1.4")
Field
Type
Description
id
ID!
GRID ID of this Player.
name
String!
Name of this Player.
participationStatus
ParticipationStatus!
Indicates the participation status of a player for this Segment.(available from version "3.8")
kills
Int!
Number of enemy kills for this Player in this Segment.
killAssistsReceived
Int!
Number of enemy kill assists received by this Player in this Segment.
killAssistsGiven
Int!
Number of enemy kill assists provided by this Player in this Segment.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists received by this Player in this Segment.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this Player in this Segment.
headshots
Int!
Number of enemy headshots for this Player.(available until version "1.9")
teamkills
Int!
Number of teamkills for this Player in this Segment.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this Player in this Segment.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this Player in this Segment.A teamkill is the occurrence of killing an allied member
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists received by this Player in this Segment.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of team kills, which were executed with a specific weapon by this Player in this Segment.
teamHeadshots
Int!
Number of allied headshots for this Player.(available until version "1.9")
selfkills
Int!
Number of selfkills (suicides) for this Player in this Segment.
deaths
Int!
Number of deaths for this Player in this Segment.
firstKill
Boolean!
Indication of player acquiring first kill.(available from version "3.10")
objectives
[Objective!]!
A list of objectives that were fulfilled by this Player in this Segment.
SegmentPlayerStateR6
R6 data points for a Player, aggregated for a Segment (available from version "1.4")
Field
Type
Description
id
ID!
GRID ID of this Player.
name
String!
Name of this Player.
participationStatus
ParticipationStatus!
Indicates the participation status of a player for this Segment.(available from version "3.8")
kills
Int!
Number of enemy kills for this Player in this Segment.
killAssistsReceived
Int!
Number of enemy kill assists received by this Player in this Segment.
killAssistsGiven
Int!
Number of enemy kill assists provided by this Player in this Segment.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists received by this Player in this Segment.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this Player in this Segment.
headshots
Int!
Number of enemy headshots for this Player.(available until version "1.9")
teamkills
Int!
Number of teamkills for this Player in this Segment.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this Player in this Segment.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this Player in this Segment.A teamkill is the occurrence of killing an allied member
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists received by this Player in this Segment.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of team kills, which were executed with a specific weapon by this Player in this Segment.
teamHeadshots
Int!
Number of allied headshots for this Player.(available until version "1.9")
selfkills
Int!
Number of selfkills (suicides) for this Player in this Segment.
deaths
Int!
Number of deaths for this Player in this Segment.
firstKill
Boolean!
Indication of player acquiring first kill.(available from version "3.10")
objectives
[Objective!]!
A list of objectives that were fulfilled by this Player in this Segment.
alive
Boolean!
Indicates whether the player is alive.
currentHealth
Int!
The current health of the player.
damageDealt
Int!
The amount of damage dealt.
damageTaken
Int!
The amount of total damage taken.
maxHealth
Int!
The max amount of health of the player.
selfdamageDealt
Int!
The amount of damage dealt to self.
selfdamageTaken
Int!
The amount of damage taken from self.
teamdamageDealt
Int!
The amount of damage dealt to team.
teamdamageTaken
Int!
The amount of damage taken from team.
healingDealt
Int!
The amount of healing dealt by team.(available from version "3.5")
healingReceived
Int!
The amount of healing received by team.(available from version "3.5")
knockdownsGiven
Int!
Number of times of knocking down an enemy for this player.(available from version "3.9")
knockdownsReceived
Int!
Number of times of being knocked down for this player.(available from version "3.9")
damageDealtSources
[DamageDealtSource!]!
A list of damage sources.(available from version "3.17")
damageDealtTargets
[DamageDealtTarget!]!
A list of damage targets.(available from version "3.17")
SegmentPlayerStateValorant
Valorant data points for a Player, aggregated for a Segment. (available from version "1.4")
Field
Type
Description
id
ID!
GRID ID of this Player.
name
String!
Name of this Player.
participationStatus
ParticipationStatus!
Indicates the participation status of a player for this Segment.(available from version "3.8")
kills
Int!
Number of enemy kills for this Player in this Segment.
killAssistsReceived
Int!
Number of enemy kill assists received by this Player in this Segment.
killAssistsGiven
Int!
Number of enemy kill assists provided by this Player in this Segment.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists received by this Player in this Segment.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this Player in this Segment.
teamkills
Int!
Number of teamkills for this Player in this Segment.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this Player in this Segment.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this Player in this Segment.A teamkill is the occurrence of killing an allied member
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists received by this Player in this Segment.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of team kills, which were executed with a specific weapon by this Player in this Segment.
selfkills
Int!
Number of selfkills (suicides) for this Player in this Segment.
deaths
Int!
Number of deaths for this Player in this Segment.
firstKill
Boolean!
Indication of player acquiring first kill.(available from version "3.10")
objectives
[Objective!]!
A list of objectives that were fulfilled by this Player in this Segment.
alive
Boolean!
Indicates whether the player is alive.(available from version "1.9")
currentArmor
Int!
The amount of current armor.(available from version "1.9")
currentHealth
Int!
The current health of the player.(available from version "1.9")
maxHealth
Int!
The max amount of health of the player.(available from version "1.9")
headshots
Int!
Number of enemy headshots for this Player.
teamHeadshots
Int!
Number of team headshots for this Player.
damageDealt
Int!
The amount of damage dealt.(available from version "3.29")
damageTaken
Int!
The amount of total damage taken.(available from version "3.29")
selfdamageDealt
Int!
The amount of damage dealt to self.(available from version "3.29")
selfdamageTaken
Int!
The amount of damage taken from self.(available from version "3.29")
teamdamageDealt
Int!
The amount of damage dealt to team.(available from version "3.29")
teamdamageTaken
Int!
The amount of damage taken from team.(available from version "3.29")
damageDealtSources
[DamageDealtSource!]!
A list of damage sources.(available from version "3.29")
damageDealtTargets
[DamageDealtTarget!]!
A list of damage targets.(available from version "3.29")
SegmentState
The state of a Segment (e.g. a round). (available from version "1.4")
Field
Type
Description
id
ID!
GRID ID of this Segment.
type
String!
The type of this Segment.
sequenceNumber
Int!
The sequence number of this Segment.
started
Boolean!
Indicates whether this Series has started.
draftActions
[DraftAction!]!
A list of DraftAction, containing information about draft actions in this Segment.(available from version "2.3")
finished
Boolean!
Indicates whether this Series has finished.
startedAt
DateTime
DateTime value when this Segment was started.(available from version "3.12")
teams
[SegmentTeamState!]!
A list of SegmentTeamState, containing information on the teams participating in this Segment.
segments
[SegmentState!]!
A list of SegmentStates, containing information of the segments in this Segment.
duration
Duration!
Duration of the Game.(available from version "3.16")
SegmentTeamStateCs2
CS2 data points for a Team, aggregated for a Segment.
Field
Type
Description
id
ID!
GRID ID of this Team.
name
String!
Name of this Team.
side
String!
Side that this Team has taken in this Segment ie Red or Blue, Terrorists or Counter-Terrorists.
won
Boolean!
Indicates whether this Team has won this Segment.
kills
Int!
Number of enemy kills for this Team in this Segment.
killAssistsReceived
Int!
Number of enemy kill assists received by this Team in this Segment.
killAssistsGiven
Int!
Number of enemy kill assists provided by this Team in this Segment.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists from a Player, received by this Team in this Segment.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this Player in this Segment.
teamkills
Int!
Number of teamkills for this Team in this Segment.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this Team in this Segment.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this Team in this Segment.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists from a Player, received by this Team in this Segment.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of team kills, which were executed with a specific weapon by this Team in this Segment.
selfkills
Int!
Number of selfkills (suicides) for this Team in this Segment.
deaths
Int!
Number of deaths for this Team in this Segment.
firstKill
Boolean!
Indication of team acquiring first kill.(available from version "3.10")
objectives
[Objective!]!
A list of objectives that were fulfilled by this Player in this Segment.
players
[SegmentPlayerState!]!
A list of SegmentPlayerState, containing information about the Players that are members of this Team.
damageDealt
Int!
The amount of damage dealt.
damageTaken
Int!
The amount of total damage taken.
selfdamageDealt
Int!
The amount of damage dealt to self.
selfdamageTaken
Int!
The amount of damage taken from self.
teamdamageDealt
Int!
The amount of damage dealt to team.
teamdamageTaken
Int!
The amount of damage taken from team.
winType
String!
If this team won the round, contains the reason for winning - otherwise an empty string.
headshots
Int!
Number of enemy headshots for this Player.
teamHeadshots
Int!
Number of allied headshots for this Player.
damageDealtSources
[DamageDealtSource!]!
A list of damage sources.(available from version "3.17")
damageDealtTargets
[DamageDealtTarget!]!
A list of damage targets.(available from version "3.17")
SegmentTeamStateCsgo
CSGO data points for a Team, aggregated for a Segment. (available from version "1.4")
Field
Type
Description
id
ID!
GRID ID of this Team.
name
String!
Name of this Team.
side
String!
Side that this Team has taken in this Segment ie Red or Blue, Terrorists or Counter-Terrorists.
won
Boolean!
Indicates whether this Team has won this Segment.
kills
Int!
Number of enemy kills for this Team in this Segment.
killAssistsReceived
Int!
Number of enemy kill assists received by this Team in this Segment.
killAssistsGiven
Int!
Number of enemy kill assists provided by this Team in this Segment.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists from a Player, received by this Team in this Segment.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this Player in this Segment.
teamkills
Int!
Number of teamkills for this Team in this Segment.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this Team in this Segment.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this Team in this Segment.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists from a Player, received by this Team in this Segment.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of team kills, which were executed with a specific weapon by this Team in this Segment.
selfkills
Int!
Number of selfkills (suicides) for this Team in this Segment.
deaths
Int!
Number of deaths for this Team in this Segment.
firstKill
Boolean!
Indication of team acquiring first kill.(available from version "3.10")
objectives
[Objective!]!
A list of objectives that were fulfilled by this Player in this Segment.
players
[SegmentPlayerState!]!
A list of SegmentPlayerState, containing information about the Players that are members of this Team.
damageDealt
Int!
The amount of damage dealt.
damageTaken
Int!
The amount of total damage taken.
selfdamageDealt
Int!
The amount of damage dealt to self.
selfdamageTaken
Int!
The amount of damage taken from self.
teamdamageDealt
Int!
The amount of damage dealt to team.
teamdamageTaken
Int!
The amount of damage taken from team.
winType
String!
If this team won the round, contains the reason for winning - otherwise an empty string.
headshots
Int!
Number of enemy headshots for this Player.
teamHeadshots
Int!
Number of allied headshots for this Player.
damageDealtSources
[DamageDealtSource!]!
A list of damage sources.(available from version "3.17")
damageDealtTargets
[DamageDealtTarget!]!
A list of damage targets.(available from version "3.17")
SegmentTeamStateDefault
(available from version "1.4")
Field
Type
Description
id
ID!
GRID ID of this Team.
name
String!
Name of this Team.
side
String!
Side that this Team has taken in this Segment ie Red or Blue, Terrorists or Counter-Terrorists.
won
Boolean!
Indicates whether this Team has won this Segment.
kills
Int!
Number of enemy kills for this Team in this Segment.
killAssistsReceived
Int!
Number of enemy kill assists received by this Team in this Segment.
killAssistsGiven
Int!
Number of enemy kill assists provided by this Team in this Segment.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists from a Player, received by this Team in this Segment.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this Player in this Segment.
headshots
Int!
Number of enemy headshots for this Player.(available until version "1.9")
teamkills
Int!
Number of teamkills for this Team in this Segment.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this Team in this Segment.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this Team in this Segment.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists from a Player, received by this Team in this Segment.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of team kills, which were executed with a specific weapon by this Team in this Segment.
teamHeadshots
Int!
Number of allied headshots for this Player.(available until version "1.9")
selfkills
Int!
Number of selfkills (suicides) for this Team in this Segment.
deaths
Int!
Number of deaths for this Team in this Segment.
firstKill
Boolean!
Indication of team acquiring first kill.(available from version "3.10")
objectives
[Objective!]!
A list of objectives that were fulfilled by this Player in this Segment.
players
[SegmentPlayerState!]!
A list of SegmentPlayerState, containing information about the Players that are members of this Team.
SegmentTeamStateR6
(available from version "1.4")
Field
Type
Description
id
ID!
GRID ID of this Team.
name
String!
Name of this Team.
side
String!
Side that this Team has taken in this Segment ie Red or Blue, Terrorists or Counter-Terrorists.
won
Boolean!
Indicates whether this Team has won this Segment.
kills
Int!
Number of enemy kills for this Team in this Segment.
killAssistsReceived
Int!
Number of enemy kill assists received by this Team in this Segment.
killAssistsGiven
Int!
Number of enemy kill assists provided by this Team in this Segment.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists from a Player, received by this Team in this Segment.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this Player in this Segment.
headshots
Int!
Number of enemy headshots for this Player.(available until version "1.9")
teamkills
Int!
Number of teamkills for this Team in this Segment.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this Team in this Segment.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this Team in this Segment.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists from a Player, received by this Team in this Segment.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of team kills, which were executed with a specific weapon by this Team in this Segment.
teamHeadshots
Int!
Number of allied headshots for this Player.(available until version "1.9")
selfkills
Int!
Number of selfkills (suicides) for this Team in this Segment.
deaths
Int!
Number of deaths for this Team in this Segment.
firstKill
Boolean!
Indication of team acquiring first kill.(available from version "3.10")
objectives
[Objective!]!
A list of objectives that were fulfilled by this Player in this Segment.
players
[SegmentPlayerState!]!
A list of SegmentPlayerState, containing information about the Players that are members of this Team.
damageDealt
Int!
The amount of damage dealt.
damageTaken
Int!
The amount of total damage taken.
selfdamageDealt
Int!
The amount of damage dealt to self.
selfdamageTaken
Int!
The amount of damage taken from self.
teamdamageDealt
Int!
The amount of damage dealt to team.
teamdamageTaken
Int!
The amount of damage taken from team.
winType
String!
If this team won the round, contains the reason for winning - otherwise an empty string.(available from version "3.4")
healingDealt
Int!
The amount of healing dealt by team.(available from version "3.5")
healingReceived
Int!
The amount of healing received by team.(available from version "3.5")
knockdownsGiven
Int!
Number of times of knocking down an enemy for this team.(available from version "3.9")
knockdownsReceived
Int!
Number of times of being knocked down for this team.(available from version "3.9")
damageDealtSources
[DamageDealtSource!]!
A list of damage sources.(available from version "3.17")
damageDealtTargets
[DamageDealtTarget!]!
A list of damage targets.(available from version "3.17")
SegmentTeamStateValorant
Valorant data points for a Team, aggregated for a Segment. (available from version "1.4")
Field
Type
Description
id
ID!
GRID ID of this Team.
name
String!
Name of this Team.
side
String!
Side that this Team has taken in this Segment ie Red or Blue, Terrorists or Counter-Terrorists.
won
Boolean!
Indicates whether this Team has won this Segment.
kills
Int!
Number of enemy kills for this Team in this Segment.
killAssistsReceived
Int!
Number of enemy kill assists received by this Team in this Segment.
killAssistsGiven
Int!
Number of enemy kill assists provided by this Team in this Segment.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists from a Player, received by this Team in this Segment.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this Player in this Segment.
teamkills
Int!
Number of teamkills for this Team in this Segment.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this Team in this Segment.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this Team in this Segment.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists from a Player, received by this Team in this Segment.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of team kills, which were executed with a specific weapon by this Team in this Segment.
selfkills
Int!
Number of selfkills (suicides) for this Team in this Segment.
deaths
Int!
Number of deaths for this Team in this Segment.
firstKill
Boolean!
Indication of team acquiring first kill.(available from version "3.10")
objectives
[Objective!]!
A list of objectives that were fulfilled by this Player in this Segment.
players
[SegmentPlayerState!]!
A list of SegmentPlayerState, containing information about the Players that are members of this Team.
winType
String!
If this team won the round, contains the reason for winning - otherwise an empty string.(available from version "1.9")
headshots
Int!
Number of enemy headshots for this Player.
teamHeadshots
Int!
Number of allied headshots for this Player.
damageDealt
Int!
The amount of damage dealt.(available from version "3.29")
damageTaken
Int!
The amount of total damage taken.(available from version "3.29")
selfdamageDealt
Int!
The amount of damage dealt to self.(available from version "3.29")
selfdamageTaken
Int!
The amount of damage taken from self.(available from version "3.29")
teamdamageDealt
Int!
The amount of damage dealt to team.(available from version "3.29")
teamdamageTaken
Int!
The amount of damage taken from team.(available from version "3.29")
damageDealtSources
[DamageDealtSource!]!
A list of damage sources.(available from version "3.29")
damageDealtTargets
[DamageDealtTarget!]!
A list of damage targets.(available from version "3.29")
SeriesPlayerStateCs2
CS2 data points for a Player, aggregated for a Series.
Field
Type
Description
id
ID!
GRID ID of this Player.
name
String!
Name of this Player.
participationStatus
ParticipationStatus!
Indicates the participation status of a player for this Series
kills
Int!
Number of enemy kills for this Player in this Series.
killAssistsReceived
Int!
Number of enemy kill assists received by this Player in this Series.
killAssistsGiven
Int!
Number of enemy kill assists provided by this Player in this Series.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists received by this Player in this Series.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this Player in this Series.
teamkills
Int!
Number of teamkills for this Player in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this Player in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this Player in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists received by this Player in this Series.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of teamkills, which were executed with a specific weapon by this Player in this Series.A teamkill is the occurrence of killing an allied member.
selfkills
Int!
Number of selfkills (suicides) for this Player in this Series.
deaths
Int!
Number of deaths for this Player in this Series.
firstKill
Boolean!
Indication of player acquiring first kill.(available from version "3.10")
structuresDestroyed
Int!
Number of structures destroyed by this Player in this Series.
structuresCaptured
Int!
Number of structures captured by this Player in this Series.
objectives
[Objective!]!
A list of objectives that were fulfilled by this Player in this Series.
multikills
[Multikill!]!
A list of multi kills that happened by this Player in this Series
headshots
Int!
Number of enemy headshots for this Player.
teamHeadshots
Int!
Number of allied headshots for this Player.
externalLinks
[ExternalLink!]!
A list of ExternalLink, containing information about links to externally provided IDs.
SeriesPlayerStateCsgo
CSGO data points for a Player, aggregated for a Series. (available from version "1.1")
Field
Type
Description
id
ID!
GRID ID of this Player.
name
String!
Name of this Player.
participationStatus
ParticipationStatus!
Indicates the participation status of a player for this Series
kills
Int!
Number of enemy kills for this Player in this Series.
killAssistsReceived
Int!
Number of enemy kill assists received by this Player in this Series.
killAssistsGiven
Int!
Number of enemy kill assists provided by this Player in this Series.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists received by this Player in this Series.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this Player in this Series.
teamkills
Int!
Number of teamkills for this Player in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this Player in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this Player in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists received by this Player in this Series.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of teamkills, which were executed with a specific weapon by this Player in this Series.A teamkill is the occurrence of killing an allied member.
selfkills
Int!
Number of selfkills (suicides) for this Player in this Series.
deaths
Int!
Number of deaths for this Player in this Series.
firstKill
Boolean!
Indication of player acquiring first kill.(available from version "3.10")
structuresDestroyed
Int!
Number of structures destroyed by this Player in this Series.
structuresCaptured
Int!
Number of structures captured by this Player in this Series.
objectives
[Objective!]!
A list of objectives that were fulfilled by this Player in this Series.
multikills
[Multikill!]!
A list of multi kills that happened by this Player in this Series(available from version "1.7")
headshots
Int!
Number of enemy headshots for this Player.
teamHeadshots
Int!
Number of allied headshots for this Player.
externalLinks
[ExternalLink!]!
A list of ExternalLink, containing information about links to externally provided IDs.(available from version "2.2")
SeriesPlayerStateDefault
Default data points for a Player, aggregated for a Series.
Field
Type
Description
id
ID!
GRID ID of this Player.
name
String!
Name of this Player.
participationStatus
ParticipationStatus!
Indicates the participation status of a player for this Series
kills
Int!
Number of enemy kills for this Player in this Series.
killAssistsReceived
Int!
Number of enemy kill assists received by this Player in this Series.
killAssistsGiven
Int!
Number of enemy kill assists provided by this Player in this Series.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists received by this Player in this Series.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this Player in this Series.
headshots
Int!
Number of enemy headshots for this Player.(available until version "1.9")
teamkills
Int!
Number of teamkills for this Player in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this Player in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this Player in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists received by this Player in this Series.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of teamkills, which were executed with a specific weapon by this Player in this Series.A teamkill is the occurrence of killing an allied member.
teamHeadshots
Int!
Number of allied headshots for this Player.(available until version "1.9")
selfkills
Int!
Number of selfkills (suicides) for this Player in this Series.
deaths
Int!
Number of deaths for this Player in this Series.
firstKill
Boolean!
Indication of player acquiring first kill.(available from version "3.10")
structuresDestroyed
Int!
Number of structures destroyed by this Player in this Series.
structuresCaptured
Int!
Number of structures captured by this Player in this Series.
objectives
[Objective!]!
A list of objectives that were fulfilled by this Player in this Series.
multikills
[Multikill!]!
A list of multi kills that happened by this Player in this Series(available from version "1.7")
SeriesPlayerStateDota
Dota data points for a Player, aggregated for a Series. (available from version "1.1")
Field
Type
Description
id
ID!
GRID ID of this Player.
name
String!
Name of this Player.
participationStatus
ParticipationStatus!
Indicates the participation status of a player for this Series
kills
Int!
Number of enemy kills for this Player in this Series.
killAssistsReceived
Int!
Number of enemy kill assists received by this Player in this Series.
killAssistsGiven
Int!
Number of enemy kill assists provided by this Player in this Series.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists received by this Player in this Series.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this Player in this Series.
headshots
Int!
Number of enemy headshots for this Player.(available until version "1.9")
teamkills
Int!
Number of teamkills for this Player in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this Player in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this Player in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists received by this Player in this Series.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of teamkills, which were executed with a specific weapon by this Player in this Series.A teamkill is the occurrence of killing an allied member.
teamHeadshots
Int!
Number of allied headshots for this Player.(available until version "1.9")
selfkills
Int!
Number of selfkills (suicides) for this Player in this Series.
deaths
Int!
Number of deaths for this Player in this Series.
firstKill
Boolean!
Indication of player acquiring first kill.(available from version "3.10")
structuresDestroyed
Int!
Number of structures destroyed by this Player in this Series.
structuresCaptured
Int!
Number of structures captured by this Player in this Series.
objectives
[Objective!]!
A list of objectives that were fulfilled by this Player in this Series.
multikills
[Multikill!]!
A list of multi kills that happened by this Player in this Series(available from version "1.7")
externalLinks
[ExternalLink!]!
A list of ExternalLink, containing information about links to externally provided IDs.(available from version "3.6")
SeriesPlayerStatePubg
PUBG data points for a Player, aggregated for a Series. (available from version "1.1")
Field
Type
Description
id
ID!
GRID ID of this Player.
name
String!
Name of this Player.
participationStatus
ParticipationStatus!
Indicates the participation status of a player for this Series
kills
Int!
Number of enemy kills for this Player in this Series.
killAssistsReceived
Int!
Number of enemy kill assists received by this Player in this Series.
killAssistsGiven
Int!
Number of enemy kill assists provided by this Player in this Series.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists received by this Player in this Series.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this Player in this Series.
teamkills
Int!
Number of teamkills for this Player in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this Player in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this Player in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists received by this Player in this Series.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of teamkills, which were executed with a specific weapon by this Player in this Series.A teamkill is the occurrence of killing an allied member.
selfkills
Int!
Number of selfkills (suicides) for this Player in this Series.
deaths
Int!
Number of deaths for this Player in this Series.
firstKill
Boolean!
Indication of player acquiring first kill.(available from version "3.10")
structuresDestroyed
Int!
Number of structures destroyed by this Player in this Series.
structuresCaptured
Int!
Number of structures captured by this Player in this Series.
objectives
[Objective!]!
A list of objectives that were fulfilled by this Player in this Series.
multikills
[Multikill!]!
A list of multi kills that happened by this Player in this Series(available from version "1.7")
headshots
Int!
Number of enemy headshots for this Player.
teamHeadshots
Int!
Number of allied headshots for this Player.
SeriesPlayerStateR6
R6 data points for a Player, aggregated for a Series
Field
Type
Description
id
ID!
GRID ID of this Player.
name
String!
Name of this Player.
participationStatus
ParticipationStatus!
Indicates the participation status of a player for this Series
kills
Int!
Number of enemy kills for this Player in this Series.
killAssistsReceived
Int!
Number of enemy kill assists received by this Player in this Series.
killAssistsGiven
Int!
Number of enemy kill assists provided by this Player in this Series.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists received by this Player in this Series.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this Player in this Series.
headshots
Int!
Number of enemy headshots for this Player.(available until version "1.9")
teamkills
Int!
Number of teamkills for this Player in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this Player in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this Player in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists received by this Player in this Series.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of teamkills, which were executed with a specific weapon by this Player in this Series.A teamkill is the occurrence of killing an allied member.
teamHeadshots
Int!
Number of allied headshots for this Player.(available until version "1.9")
selfkills
Int!
Number of selfkills (suicides) for this Player in this Series.
deaths
Int!
Number of deaths for this Player in this Series.
firstKill
Boolean!
Indication of player acquiring first kill.(available from version "3.10")
structuresDestroyed
Int!
Number of structures destroyed by this Player in this Series.
structuresCaptured
Int!
Number of structures captured by this Player in this Series.
objectives
[Objective!]!
A list of objectives that were fulfilled by this Player in this Series.
multikills
[Multikill!]!
A list of multi kills that happened by this Player in this Series(available from version "1.7")
knockdownsGiven
Int!
Number of times of knocking down an enemy for this player.(available from version "3.9")
knockdownsReceived
Int!
Number of times of being knocked down for this player.(available from version "3.9")
SeriesPlayerStateValorant
Valorant data points for a Player, aggregated for a Series. (available from version "1.1")
Field
Type
Description
id
ID!
GRID ID of this Player.
name
String!
Name of this Player.
participationStatus
ParticipationStatus!
Indicates the participation status of a player for this Series
kills
Int!
Number of enemy kills for this Player in this Series.
killAssistsReceived
Int!
Number of enemy kill assists received by this Player in this Series.
killAssistsGiven
Int!
Number of enemy kill assists provided by this Player in this Series.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists received by this Player in this Series.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this Player in this Series.
teamkills
Int!
Number of teamkills for this Player in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this Player in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this Player in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists received by this Player in this Series.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of teamkills, which were executed with a specific weapon by this Player in this Series.A teamkill is the occurrence of killing an allied member.
selfkills
Int!
Number of selfkills (suicides) for this Player in this Series.
deaths
Int!
Number of deaths for this Player in this Series.
firstKill
Boolean!
Indication of player acquiring first kill.(available from version "3.10")
structuresDestroyed
Int!
Number of structures destroyed by this Player in this Series.
structuresCaptured
Int!
Number of structures captured by this Player in this Series.
objectives
[Objective!]!
A list of objectives that were fulfilled by this Player in this Series.
multikills
[Multikill!]!
A list of multi kills that happened by this Player in this Series(available from version "1.7")
headshots
Int!
Number of enemy headshots for this Player.
teamHeadshots
Int!
Number of team headshots for this Player.
SeriesState
Data points for a Series.
Field
Type
Description
version
Version!
Version of the data model returned.
id
ID!
GRID ID of this Series.
title
Title!
Esports Title of this Series.(available from version "1.1")
format
String!
Format of this Series e.g. Best of 3 (Bo3).
started
Boolean!
Indicates whether this Series has started.
finished
Boolean!
Indicates whether this Series has finished.
forfeited
Boolean!
Indicates whether this Series was forfeited.(available from version "3.48")
valid
Boolean!
Indicates whether this Series data is considered accurate.
teams
[SeriesTeamState!]!
A list of SeriesTeamState, containing information about the Teams participating in this Series.
games
[GameState!]!
A list of GameState, containing information about the Games in this Series
draftActions
[DraftAction!]!
A list of DraftAction, containing information about draft actions in this Series.
updatedAt
DateTime!
DateTime value when this Series data was last updated.
startedAt
DateTime
DateTime value when this Series was started.(available from version "2.1")
duration
Duration!
Duration of the whole Series.(available from version "3.14")
SeriesTeamStateCs2
CS2 data points for a Team, aggregated for a Series.
Field
Type
Description
id
ID!
GRID ID of this team..
name
String!
Name of this team.
score
Int!
Score for this team in this Series. This is used for the Score After format and Best-of (#games won).
won
Boolean!
Indicates whether this team has won this Series.
kills
Int!
Number of enemy kills for this team in this Series.
killAssistsReceived
Int!
Number of enemy kill assists received by this team in this Series.
killAssistsGiven
Int!
Number of enemy kill assists provided by this team in this Series.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists from a Player, received by this Team in this Series.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this team in this Series.
teamkills
Int!
Number of teamkills for this team in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this team in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this team in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists from a Player, received by this Team in this Series.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of teamkills, which were executed with a specific weapon by this team in this Series.A teamkill is the occurrence of killing an allied member.
selfkills
Int!
Number of selfkills (suicides) for this team in this Series.
deaths
Int!
Number of deaths for this team in this Series.
firstKill
Boolean!
Indication of team acquiring first kill.(available from version "3.10")
structuresDestroyed
Int!
Number of structures destroyed by this team in this Series.
structuresCaptured
Int!
Number of structures captured by this team in this Series.
objectives
[Objective!]!
A list of objectives that were fulfilled by this team in this Series.
players
[SeriesPlayerState!]!
A list of SeriesPlayerState, containing information about the Players that are members of this team.
multikills
[Multikill!]!
A list of multi kills that happened by this Team in this Series.(available from version "1.7")
headshots
Int!
Number of enemy headshots for this Player.
teamHeadshots
Int!
Number of allied headshots for this Player.
SeriesTeamStateCsgo
CSGO data points for a Team, aggregated for a Series. (available from version "1.1")
Field
Type
Description
id
ID!
GRID ID of this team..
name
String!
Name of this team.
score
Int!
Score for this team in this Series. This is used for the Score After format and Best-of (#games won).
won
Boolean!
Indicates whether this team has won this Series.
kills
Int!
Number of enemy kills for this team in this Series.
killAssistsReceived
Int!
Number of enemy kill assists received by this team in this Series.
killAssistsGiven
Int!
Number of enemy kill assists provided by this team in this Series.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists from a Player, received by this Team in this Series.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this team in this Series.
teamkills
Int!
Number of teamkills for this team in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this team in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this team in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists from a Player, received by this Team in this Series.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of teamkills, which were executed with a specific weapon by this team in this Series.A teamkill is the occurrence of killing an allied member.
selfkills
Int!
Number of selfkills (suicides) for this team in this Series.
deaths
Int!
Number of deaths for this team in this Series.
firstKill
Boolean!
Indication of team acquiring first kill.(available from version "3.10")
structuresDestroyed
Int!
Number of structures destroyed by this team in this Series.
structuresCaptured
Int!
Number of structures captured by this team in this Series.
objectives
[Objective!]!
A list of objectives that were fulfilled by this team in this Series.
players
[SeriesPlayerState!]!
A list of SeriesPlayerState, containing information about the Players that are members of this team.
multikills
[Multikill!]!
A list of multi kills that happened by this Team in this Series.(available from version "1.7")
headshots
Int!
Number of enemy headshots for this Player.
teamHeadshots
Int!
Number of allied headshots for this Player.
SeriesTeamStateDefault
THESE SHOULD BE IDENTICAL WITH INTERFACES
Field
Type
Description
id
ID!
GRID ID of this team..
name
String!
Name of this team.
score
Int!
Score for this team in this Series. This is used for the Score After format and Best-of (#games won).
won
Boolean!
Indicates whether this team has won this Series.
kills
Int!
Number of enemy kills for this team in this Series.
killAssistsReceived
Int!
Number of enemy kill assists received by this team in this Series.
killAssistsGiven
Int!
Number of enemy kill assists provided by this team in this Series.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists from a Player, received by this Team in this Series.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this team in this Series.
headshots
Int!
Number of enemy headshots for this Player.(available until version "1.9")
teamkills
Int!
Number of teamkills for this team in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this team in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this team in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists from a Player, received by this Team in this Series.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of teamkills, which were executed with a specific weapon by this team in this Series.A teamkill is the occurrence of killing an allied member.
teamHeadshots
Int!
Number of allied headshots for this Player.(available until version "1.9")
selfkills
Int!
Number of selfkills (suicides) for this team in this Series.
deaths
Int!
Number of deaths for this team in this Series.
firstKill
Boolean!
Indication of team acquiring first kill.(available from version "3.10")
structuresDestroyed
Int!
Number of structures destroyed by this team in this Series.
structuresCaptured
Int!
Number of structures captured by this team in this Series.
objectives
[Objective!]!
A list of objectives that were fulfilled by this team in this Series.
players
[SeriesPlayerState!]!
A list of SeriesPlayerState, containing information about the Players that are members of this team.
multikills
[Multikill!]!
A list of multi kills that happened by this Team in this Series.(available from version "1.7")
SeriesTeamStateDota
Dota data points for a Team, aggregated for a Series. (available from version "1.1")
Field
Type
Description
id
ID!
GRID ID of this team..
name
String!
Name of this team.
score
Int!
Score for this team in this Series. This is used for the Score After format and Best-of (#games won).
won
Boolean!
Indicates whether this team has won this Series.
kills
Int!
Number of enemy kills for this team in this Series.
killAssistsReceived
Int!
Number of enemy kill assists received by this team in this Series.
killAssistsGiven
Int!
Number of enemy kill assists provided by this team in this Series.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists from a Player, received by this Team in this Series.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this team in this Series.
headshots
Int!
Number of enemy headshots for this Player.(available until version "1.9")
teamkills
Int!
Number of teamkills for this team in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this team in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this team in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists from a Player, received by this Team in this Series.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of teamkills, which were executed with a specific weapon by this team in this Series.A teamkill is the occurrence of killing an allied member.
teamHeadshots
Int!
Number of allied headshots for this Player.(available until version "1.9")
selfkills
Int!
Number of selfkills (suicides) for this team in this Series.
deaths
Int!
Number of deaths for this team in this Series.
firstKill
Boolean!
Indication of team acquiring first kill.(available from version "3.10")
structuresDestroyed
Int!
Number of structures destroyed by this team in this Series.
structuresCaptured
Int!
Number of structures captured by this team in this Series.
objectives
[Objective!]!
A list of objectives that were fulfilled by this team in this Series.
players
[SeriesPlayerState!]!
A list of SeriesPlayerState, containing information about the Players that are members of this team.
multikills
[Multikill!]!
A list of multi kills that happened by this Team in this Series.(available from version "1.7")
SeriesTeamStatePubg
PUBG data points for a Team, aggregated for a Series. (available from version "1.1")
Field
Type
Description
id
ID!
GRID ID of this team..
name
String!
Name of this team.
score
Int!
Score for this team in this Series. This is used for the Score After format and Best-of (#games won).
won
Boolean!
Indicates whether this team has won this Series.
kills
Int!
Number of enemy kills for this team in this Series.
killAssistsReceived
Int!
Number of enemy kill assists received by this team in this Series.
killAssistsGiven
Int!
Number of enemy kill assists provided by this team in this Series.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists from a Player, received by this Team in this Series.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this team in this Series.
teamkills
Int!
Number of teamkills for this team in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this team in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this team in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists from a Player, received by this Team in this Series.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of teamkills, which were executed with a specific weapon by this team in this Series.A teamkill is the occurrence of killing an allied member.
selfkills
Int!
Number of selfkills (suicides) for this team in this Series.
deaths
Int!
Number of deaths for this team in this Series.
firstKill
Boolean!
Indication of team acquiring first kill.(available from version "3.10")
structuresDestroyed
Int!
Number of structures destroyed by this team in this Series.
structuresCaptured
Int!
Number of structures captured by this team in this Series.
objectives
[Objective!]!
A list of objectives that were fulfilled by this team in this Series.
players
[SeriesPlayerState!]!
A list of SeriesPlayerState, containing information about the Players that are members of this team.
multikills
[Multikill!]!
A list of multi kills that happened by this Team in this Series.(available from version "1.7")
headshots
Int!
Number of enemy headshots for this Player.
teamHeadshots
Int!
Number of allied headshots for this Player.
SeriesTeamStateR6
Field
Type
Description
id
ID!
GRID ID of this team..
name
String!
Name of this team.
score
Int!
Score for this team in this Series. This is used for the Score After format and Best-of (#games won).
won
Boolean!
Indicates whether this team has won this Series.
kills
Int!
Number of enemy kills for this team in this Series.
killAssistsReceived
Int!
Number of enemy kill assists received by this team in this Series.
killAssistsGiven
Int!
Number of enemy kill assists provided by this team in this Series.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists from a Player, received by this Team in this Series.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this team in this Series.
headshots
Int!
Number of enemy headshots for this Player.(available until version "1.9")
teamkills
Int!
Number of teamkills for this team in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this team in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this team in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists from a Player, received by this Team in this Series.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of teamkills, which were executed with a specific weapon by this team in this Series.A teamkill is the occurrence of killing an allied member.
teamHeadshots
Int!
Number of allied headshots for this Player.(available until version "1.9")
selfkills
Int!
Number of selfkills (suicides) for this team in this Series.
deaths
Int!
Number of deaths for this team in this Series.
firstKill
Boolean!
Indication of player acquiring first kill.(available from version "3.10")
structuresDestroyed
Int!
Number of structures destroyed by this team in this Series.
structuresCaptured
Int!
Number of structures captured by this team in this Series.
objectives
[Objective!]!
A list of objectives that were fulfilled by this team in this Series.
players
[SeriesPlayerState!]!
A list of SeriesPlayerState, containing information about the Players that are members of this team.
multikills
[Multikill!]!
A list of multi kills that happened by this Team in this Series.(available from version "1.7")
knockdownsGiven
Int!
Number of times of knocking down an enemy for this team.(available from version "3.9")
knockdownsReceived
Int!
Number of times of being knocked down for this team.(available from version "3.9")
SeriesTeamStateValorant
Valorant data points for a Team, aggregated for a Series. (available from version "1.1")
Field
Type
Description
id
ID!
### BASE FIELDS ####
name
String!
Name of this team.
score
Int!
Score for this team in this Series. This is used for the Score After format and Best-of (#games won).
won
Boolean!
Indicates whether this team has won this Series.
kills
Int!
Number of enemy kills for this team in this Series.
killAssistsReceived
Int!
Number of enemy kill assists received by this team in this Series.
killAssistsGiven
Int!
Number of enemy kill assists provided by this team in this Series.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists from a Player, received by this Team in this Series.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this team in this Series.
teamkills
Int!
Number of teamkills for this team in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this team in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this team in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists from a Player, received by this Team in this Series.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of teamkills, which were executed with a specific weapon by this team in this Series.A teamkill is the occurrence of killing an allied member.
selfkills
Int!
Number of selfkills (suicides) for this team in this Series.
deaths
Int!
Number of deaths for this team in this Series.
firstKill
Boolean!
Indication of team acquiring first kill.(available from version "3.10")
structuresDestroyed
Int!
Number of structures destroyed by this team in this Series.
structuresCaptured
Int!
Number of structures captured by this team in this Series.
objectives
[Objective!]!
A list of objectives that were fulfilled by this team in this Series.
players
[SeriesPlayerState!]!
A list of SeriesPlayerState, containing information about the Players that are members of this team.
multikills
[Multikill!]!
A list of multi kills that happened by this Team in this Series.(available from version "1.7")
headshots
Int!
Number of enemy headshots for this Player.
teamHeadshots
Int!
Number of allied headshots for this Player.
StatusEffect
A status effect affecting a player. (available from version "3.31")
Field
Type
Description
id
ID!
GRID ID for this effect.
name
String!
The name of this effect.
quantity
Int!
The stack count of this buff.(available from version "3.39")
clock
ClockState
Clock state to indicate when the effect expires.
StructureState
Data points for Structures in a Game.
Field
Type
Description
id
ID!
GRID Structure ID.
type
String!
Type of the Structure.
side
String!
Side that controls the Structure.
teamId
ID!
GRID Team ID that controls the Structure.
respawnClock
ClockState
Respawn clock state to indicate when a structure respawns.(available from version "3.19")
currentHealth
Int!
The current health of the Structure.(available from version "1.6")
maxHealth
Int!
The max amount of health of the Structure.(available from version "1.6")
destroyed
Boolean!
Indicates whether the structure has been destroyed.
position
Coordinates
Structure Coordinates on the map.
TeamkillAssistFromPlayer
Teamkill assist information.
Field
Type
Description
id
ID!
GRID ID for this assist.
playerId
ID!
GRID Player ID assisting.
teamkillAssistsReceived
Int!
Number of teamkill assists received from the assisting player.
Title
An esports Title. (available from version "1.1")
Field
Type
Description
nameShortened
String!
Unique, short name description of the esports Title.
TitleVersion
Title version information
Field
Type
Description
name
String!
Version name
UnitKill
A unit kill caused by a player or a team. (available from version "3.1")
Field
Type
Description
id
ID!
GRID ID for this unit kill
unitName
String!
The name of unit that got killed
count
Int!
Amount of times a specific unit was killed
WeaponKill
A kill that was executed with the named weapon.
Field
Type
Description
id
ID!
GRID ID for this weapon kill.
weaponName
String!
Name of the weapon used for this kill.
count
Int
Amount of times a kill happened with the named weapon.
Inputs
GameStateFilter
Generic filter that can be used to query for Games that have/haven’t started and are/aren’t finished.
Field
Type
Description
started
Boolean
Filter on whether a Game has started.
finished
Boolean
Filter on whether a Game has finished.
Interfaces
Ability
An ability that is owned by a player. (available from version "1.8")
Field
Type
Description
id
ID!
GRID ID for this ability.
name
String!
The name of this ability.
ready
Boolean!
Whether this ability can be activated or not.
GamePlayerState
Data points for a Player, aggregated for a Game
Field
Type
Description
id
ID!
GRID ID of this Player.
name
String!
Name of this Player.
character
Character
In-game character (ie champion, class) of the Player in this Game
roles
[PlayerRole!]!
Roles this Player has in this Game.(available from version "3.43")
participationStatus
ParticipationStatus!
Indicates the participation status of a player for this Game
money
Int!
Amount of money this Player is holding in cash.Depending on the Title this may represent in-game gold, credits, gems etc.
loadoutValue
Int!
Total value (worth in money) of this Player’s loadout.
netWorth
Int!
Sum of money and inventoryValue for this Player.
kills
Int!
Number of enemy kills for this Player in this Game.
killAssistsReceived
Int!
Number of enemy kill assists received by this Player in this Game.
killAssistsGiven
Int!
Number of enemy kill assists provided by this Player in this Game.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists received by this Player in this Game.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this Player in this Game.
teamkills
Int!
Number of teamkills for this Player in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this Player in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this Player in this Game.A teamkill is the occurrence of killing an allied member
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists received by this Player in this Game.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of teamkills, which were executed with a specific weapon by this Player in this Game.A teamkill is the occurrence of killing an allied member.
selfkills
Int!
Number of selfkills (suicides) for this Player in this Game.
deaths
Int!
Number of deaths for this Player in this Game.
firstKill
Boolean!
Indication of player acquiring first kill.(available from version "3.10")
structuresDestroyed
Int!
Number of structures destroyed by this Player in this Game.
structuresCaptured
Int!
Number of structures captured by this Player in this Game.
inventory
PlayerInventory!
The Player's inventory
objectives
[Objective!]!
A list of objectives that were fulfilled by this Player in this Game.
position
Coordinates
Player Coordinates on the map
multikills
[Multikill!]!
A list of multi kills that happened by this Player in this Game.(available from version "1.7")
unitKills
[UnitKill!]!
A list of unit kills that happened by this Player in this Game.(available from version "3.1")
abilities
[Ability!]!
A list of abilities that are owned by this Player in this Game.(available from version "1.8")
statusEffects
[StatusEffect!]!
A list of status effects affecting the player(available from version "3.31")
GameTeamState
Data points for a Team, aggregated for a Game.
Field
Type
Description
id
ID!
GRID ID of this Team.
name
String!
Name of this Team.
side
String!
Side that this Team has taken in this Game ie Red or Blue, Terrorists or Counter-Terrorists.
won
Boolean!
Indicates whether this Team has won this Game.
score
Int!
Score for this Team in this Game. This is used in the Score After format.
money
Int!
Amount of money this Team is holding in cash.Depending on the Title this may represent in-game gold, credits, gems etc.
loadoutValue
Int!
Total value (worth in money) of all the loadouts owned by Players that are members of this Team.
netWorth
Int!
Sum of money and inventoryValue for this Team.
kills
Int!
Number of enemy kills for this Team in this Game.
killAssistsReceived
Int!
Number of enemy kill assists received by this Team in this Game.
killAssistsGiven
Int!
Number of enemy kill assists provided by this Team in this Game.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists from a Player, received by this Team in this Game.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this Team in this Game.
teamkills
Int!
Number of teamkills for this Team in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this Team in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this Team in this Game.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists from a Player, received by this Team in this Game.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of teamkills, which were executed with a specific weapon by this Team in this Game.A teamkill is the occurrence of killing an allied member.
selfkills
Int!
Number of selfkills (suicides) for this Team in this Game.
deaths
Int!
Number of deaths for this Team in this Game.
firstKill
Boolean!
Indication of team acquiring first kill.(available from version "3.10")
structuresDestroyed
Int!
Number of structures destroyed by this Team in this Game.
structuresCaptured
Int!
Number of structures captured by this Team in this Game.
objectives
[Objective!]!
A list of objectives that were fulfilled by this Team in this Game.
unitKills
[UnitKill!]!
A list of unit kills that happened by this Team in this Game.(available from version "3.1")
players
[GamePlayerState!]!
A list of GamePlayerState, containing information about the Players that are members of this Team.
multikills
[Multikill!]!
A list of multi kills that happened by this Team in this Game.(available from version "1.7")
SegmentPlayerState
Data points for a Player, aggregated for a Segment (available from version "1.4")
Field
Type
Description
id
ID!
GRID ID of this Player.
name
String!
Name of this Player.
participationStatus
ParticipationStatus!
Indicates the participation status of a player for this Segment.(available from version "3.8")
kills
Int!
Number of enemy kills for this Player in this Segment.
killAssistsReceived
Int!
Number of enemy kill assists received by this Player in this Segment.
killAssistsGiven
Int!
Number of enemy kill assists provided by this Player in this Segment.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists received by this Player in this Segment.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this Player in this Segment.
teamkills
Int!
Number of teamkills for this Player in this Segment.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this Player in this Segment.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this Player in this Segment.A teamkill is the occurrence of killing an allied member
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists received by this Player in this Segment.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of team kills, which were executed with a specific weapon by this Player in this Segment.
selfkills
Int!
Number of selfkills (suicides) for this Player in this Segment.
deaths
Int!
Number of deaths for this Player in this Segment.
firstKill
Boolean!
Indication of player acquiring first kill.(available from version "3.10")
objectives
[Objective!]!
A list of objectives that were fulfilled by this Player in this Segment.
SegmentTeamState
Data points for a Team, aggregated for a Segment. (available from version "1.4")
Field
Type
Description
id
ID!
GRID ID of this Team.
name
String!
Name of this Team.
side
String!
Side that this Team has taken in this Segment ie Red or Blue, Terrorists or Counter-Terrorists.
won
Boolean!
Indicates whether this Team has won this Segment.
kills
Int!
Number of enemy kills for this Team in this Segment.
killAssistsReceived
Int!
Number of enemy kill assists received by this Team in this Segment.
killAssistsGiven
Int!
Number of enemy kill assists provided by this Team in this Segment.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists from a Player, received by this Team in this Segment.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this Player in this Segment.
teamkills
Int!
Number of teamkills for this Team in this Segment.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this Team in this Segment.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this Team in this Segment.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists from a Player, received by this Team in this Segment.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of team kills, which were executed with a specific weapon by this Team in this Segment.
selfkills
Int!
Number of selfkills (suicides) for this Team in this Segment.
deaths
Int!
Number of deaths for this Team in this Segment.
firstKill
Boolean!
Indication of team acquiring first kill.(available from version "3.10")
objectives
[Objective!]!
A list of objectives that were fulfilled by this Player in this Segment.
players
[SegmentPlayerState!]!
A list of SegmentPlayerState, containing information about the Players that are members of this Team.
SeriesPlayerState
Data points for a Player, aggregated for a Series.
Field
Type
Description
id
ID!
GRID ID of this Player.
name
String!
Name of this Player.
participationStatus
ParticipationStatus!
Indicates the participation status of a player for this Series
kills
Int!
Number of enemy kills for this Player in this Series.
killAssistsReceived
Int!
Number of enemy kill assists received by this Player in this Series.
killAssistsGiven
Int!
Number of enemy kill assists provided by this Player in this Series.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists received by this Player in this Series.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this Player in this Series.
teamkills
Int!
Number of teamkills for this Player in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this Player in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this Player in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists received by this Player in this Series.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of teamkills, which were executed with a specific weapon by this Player in this Series.A teamkill is the occurrence of killing an allied member.
selfkills
Int!
Number of selfkills (suicides) for this Player in this Series.
deaths
Int!
Number of deaths for this Player in this Series.
firstKill
Boolean!
Indication of player acquiring first kill.(available from version "3.10")
structuresDestroyed
Int!
Number of structures destroyed by this Player in this Series.
structuresCaptured
Int!
Number of structures captured by this Player in this Series.
objectives
[Objective!]!
A list of objectives that were fulfilled by this Player in this Series.
multikills
[Multikill!]!
A list of multi kills that happened by this Player in this Series(available from version "1.7")
SeriesTeamState
Data points for a Team, aggregated for a Series.
Field
Type
Description
id
ID!
GRID ID of this team..
name
String!
Name of this team.
score
Int!
Score for this team in this Series. This is used for the Score After format and Best-of (#games won).
won
Boolean!
Indicates whether this team has won this Series.
kills
Int!
Number of enemy kills for this team in this Series.
killAssistsReceived
Int!
Number of enemy kill assists received by this team in this Series.
killAssistsGiven
Int!
Number of enemy kill assists provided by this team in this Series.
killAssistsReceivedFromPlayer
[KillAssistFromPlayer!]!
A list of enemy KillAssistFromPlayer, containing information on kill assists from a Player, received by this Team in this Series.
weaponKills
[WeaponKill!]!
A list of enemy kills, which were executed with a specific weapon by this team in this Series.
teamkills
Int!
Number of teamkills for this team in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceived
Int!
Number of teamkill assists received by this team in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsGiven
Int!
Number of teamkill assists provided by this team in this Series.A teamkill is the occurrence of killing an allied member.
teamkillAssistsReceivedFromPlayer
[TeamkillAssistFromPlayer!]!
A list of KillAssistFromPlayer, containing information on teamkill assists from a Player, received by this Team in this Series.A teamkill is the occurrence of killing an allied member.
weaponTeamkills
[WeaponKill!]!
A list of teamkills, which were executed with a specific weapon by this team in this Series.A teamkill is the occurrence of killing an allied member.
selfkills
Int!
Number of selfkills (suicides) for this team in this Series.
deaths
Int!
Number of deaths for this team in this Series.
firstKill
Boolean!
Indication of team acquiring first kill.(available from version "3.10")
structuresDestroyed
Int!
Number of structures destroyed by this team in this Series.
structuresCaptured
Int!
Number of structures captured by this team in this Series.
objectives
[Objective!]!
A list of objectives that were fulfilled by this team in this Series.
players
[SeriesPlayerState!]!
A list of SeriesPlayerState, containing information about the Players that are members of this team.
multikills
[Multikill!]!
A list of multi kills that happened by this Team in this Series.(available from version "1.7")
Enums
ErrorDetail
Field
Description
UNKNOWN
Unknown error. This error should only be returned when no other error detail applies. If a client sees an unknown errorDetail, it will be interpreted as UNKNOWN. HTTP Mapping: 500 Internal Server Error
FIELD_NOT_FOUND
The requested field is not found in the schema. This differs from
INVALID_CURSOR
The provided cursor is not valid. The most common usage for this error is when a client is paginating through a list that uses stateful cursors. In that case, the provided cursor may be expired. HTTP Mapping: 404 Not Found Error Type: NOT_FOUND
UNIMPLEMENTED
The operation is not implemented or is not currently supported/enabled. HTTP Mapping: 501 Not Implemented Error Type: BAD_REQUEST
INVALID_ARGUMENT
The client specified an invalid argument. Note that this differs from
DEADLINE_EXCEEDED
The deadline expired before the operation could complete. For operations that change the state of the system, this error may be returned even if the operation has completed successfully. For example, a successful response from a server could have been delayed long enough for the deadline to expire. HTTP Mapping: 504 Gateway Timeout Error Type: UNAVAILABLE
SERVICE_ERROR
Service Error. There is a problem with an upstream service. This may be returned if a gateway receives an unknown error from a service or if a service is unreachable. If a request times out which waiting on a response from a service,
THROTTLED_CPU
Request throttled based on server CPU limits HTTP Mapping: 503 Unavailable. Error Type: UNAVAILABLE
THROTTLED_CONCURRENCY
Request throttled based on server concurrency limits. HTTP Mapping: 503 Unavailable Error Type: UNAVAILABLE
ENHANCE_YOUR_CALM
The server detected that the client is exhibiting a behavior that might be generating excessive load. HTTP Mapping: 420 Enhance Your Calm Error Type: UNAVAILABLE
TOO_MANY_REQUESTS
The server detected that the client is exhibiting a behavior that might be generating excessive load. HTTP Mapping: 429 Too Many Requests Error Type: UNAVAILABLE
TCP_FAILURE
Request failed due to network errors. HTTP Mapping: 503 Unavailable Error Type: UNAVAILABLE
MISSING_RESOURCE
Unable to perform operation because a required resource is missing. Example: Client is attempting to refresh a list, but the specified list is expired. This requires an action by the client to get a new list. If the user is simply trying GET a resource that is not found, use the NOT_FOUND error type. FAILED_PRECONDITION.MISSING_RESOURCE is to be used particularly when the user is performing an operation that requires a particular resource to exist. HTTP Mapping: 400 Bad Request or 500 Internal Server Error Error Type: FAILED_PRECONDITION
ErrorType
Field
Description
UNKNOWN
Unknown error. For example, this error may be returned when an error code received from another address space belongs to an error space that is not known in this address space. Also errors raised by APIs that do not return enough error information may be converted to this error. If a client sees an unknown errorType, it will be interpreted as UNKNOWN. Unknown errors MUST NOT trigger any special behavior. These MAY be treated by an implementation as being equivalent to INTERNAL. When possible, a more specific error should be provided. HTTP Mapping: 520 Unknown Error
INTERNAL
Internal error. An unexpected internal error was encountered. This means that some invariants expected by the underlying system have been broken. This error code is reserved for serious errors. HTTP Mapping: 500 Internal Server Error
NOT_FOUND
The requested entity was not found. This could apply to a resource that has never existed (e.g. bad resource id), or a resource that no longer exists (e.g. cache expired.) Note to server developers: if a request is denied for an entire class of users, such as gradual feature rollout or undocumented allowlist,
UNAUTHENTICATED
The request does not have valid authentication credentials. This is intended to be returned only for routes that require authentication. HTTP Mapping: 401 Unauthorized
PERMISSION_DENIED
The caller does not have permission to execute the specified operation.
BAD_REQUEST
Bad Request. There is a problem with the request. Retrying the same request is not likely to succeed. An example would be a query or argument that cannot be deserialized. HTTP Mapping: 400 Bad Request
UNAVAILABLE
Currently Unavailable. The service is currently unavailable. This is most likely a transient condition, which can be corrected by retrying with a backoff. HTTP Mapping: 503 Unavailable
FAILED_PRECONDITION
The operation was rejected because the system is not in a state required for the operation's execution. For example, the directory to be deleted is non-empty, an rmdir operation is applied to a non-directory, etc. Service implementers can use the following guidelines to decide between
GameType
The type of game.
Field
Description
SCRIM
A practice competitive game.
ESPORTS
An esports game.
COMPETITIVE
A competitive non-esports series.
REGULAR
A regular non-competitive game.
ParticipationStatus
Participation status of an entity (ie Player).
Field
Description
active
Entity (ie Player) actively participating.
inactive
Entity (ie Player) not actively participating anymore.
Scalars
Boolean
The 
Boolean
 scalar type represents 
true
 or 
false
.
DateTime
DateTime formatted as ISO 8601
Duration
Duration formatted as ISO 8601
Float
The 
Float
 scalar type represents signed double-precision fractional values as specified by IEEE 754.
ID
The 
ID
 scalar type represents a unique identifier, often used to refetch an object or as key for a cache. The ID type appears in a JSON response as a String; however, it is not intended to be human-readable. When expected as an input type, any string (such as 
"4"
) or integer (such as 
4
) input value will be accepted as an ID.
Int
The 
Int
 scalar type represents non-fractional signed whole numeric values. Int can represent values between -(2^31) and 2^31 - 1.
String
The 
String
 scalar type represents textual data, represented as UTF-8 character sequences. The String type is most often used by GraphQL to represent free-form human-readable text.
Version
Version represented as string using format 'MAJOR.MINO