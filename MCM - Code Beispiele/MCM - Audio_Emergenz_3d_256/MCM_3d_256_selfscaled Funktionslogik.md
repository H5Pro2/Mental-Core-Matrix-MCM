# MCM_3d_256_selfscaled Funktionslogik
## Dokumentstatus

- **Status:** Experimentelle Umsetzung oder Ergebnisdokumentation
- **Funktion:** Konkrete Übersetzung einzelner MCM-Ideen in Code, Simulation oder Prototyp
- **Geltungsbereich:** Die beschriebenen Beobachtungen gelten für die jeweilige Implementierung und sind kein Beleg für die MCM als allgemeine Theorie.
- **Interpretation:** Begriffe wie Feld, Emergenz, Gedächtnis oder Kognition bezeichnen hier Modellfunktionen. Sie behaupten keine biologischen oder bewussten Eigenschaften.



Technische Umsetzung und Funktionslogik des Skripts
1. Struktur des Systems
Das Skript implementiert ein dreidimensionales Mehrkörpersystem mit insgesamt 256 Agenten.
Die Zahl ergibt sich aus einer systematischen Aufteilung:
• 4 Gruppen
• 4 radiale Energiestufen
• 4 Phasenlagen in z-Richtung
• 4 Winkelunterteilungen pro Gruppe
Diese Kombination erzeugt ein regulär strukturiertes 3D-Energiefeld mit fester räumlicher
Geometrie.
Jeder Agent besitzt:
• eine feste Position im 3D-Raum
• einen dynamischen Energiezustand im Bereich [−3, +3]

2. Energieinitialisierung
Die Startenergien werden nicht zufällig gesetzt, sondern anhand interner Energiebänder, die den
vier Gruppen zugeordnet sind.
Jede Gruppe erhält ein eigenes Energieintervall, das über die vier radialen Ebenen systematisch
durchlaufen wird.
Damit entsteht ein geordnetes Anfangsfeld, das energetisch vorstrukturiert ist.

3. Lokale Energiekopplung
Der zentrale Mechanismus des Skripts ist die lokale Kopplung:
• jeder Agent überprüft seine Nachbarn
• nur Nachbarn innerhalb eines festen Radius interagieren
• die Energie eines Agenten wird zum Mittelwert der Nachbarn gezogen
• die Stärke dieses Angleichens wird durch einen festen Kopplungsfaktor bestimmt
Der Code verwendet dafür keine NxN-Matrix, sondern berechnet Abstände pro Agent effizient
mit Vektordifferenzen.
Die Interaktion ist rein lokal; es existiert keine globale Führung oder zentrale Steuerlogik.

4. Stochastische Energiefluktuation
Zusätzlich zur lokalen Kopplung wird jedem Agenten pro Zeitschritt ein Rauschterm hinzugefügt:
• Gaußsche Zufallswerte
• skaliert mit sqrt(dt)
• anschließend Clipping in den Bereich [−3, +3]


Das erzeugt kontinuierliche Mikrofluktuationen, die das System permanent herausfordern und
Reorganisation erzwingen.

5. Selbstskalierende Zeiteinheit
Das Modell besitzt keinen festen Zeitschritt.
Stattdessen wird dt dynamisch gesetzt:
dt = DT_MAX / (1 + BETA · Aktivität)
Die Aktivität wird als mittlere Energiedifferenz zwischen aktuellem und vorherigem Schritt
berechnet.
Daraus folgt:
• hohe Aktivität → kleines dt → Zeit verlangsamt sich
• geringe Aktivität → großes dt → Zeit beschleunigt sich
Die Simulation enthält damit eine Eigenzeit, die direkt vom Energiefluss erzeugt wird.

6. Iterative Hauptsimulation
Die Simulation führt bis zu 4000 Schritte aus.
Pro Schritt werden:
1. aktuelle Aktivität gemessen
2. dt berechnet
3. Energien der Agenten aktualisiert
4. Zeit inkrementiert
5. Energiefeld gespeichert
Resultat:
• vollständiger Zeitverlauf der Eigenzeit
• Zeitreihe aller 256 Energieverläufe
• die festen Positionen aller Agenten

7. Energie-zu-Frequenz-Mapping
Zur akustischen Umsetzung wird jede Energie E ∈ [−3, +3] linear auf einen Frequenzraum
abgebildet:
• unteren Grenze: 90 Hz
• oberen Grenze: 1200 Hz
Damit entsteht ein 256-stimmiges Tonsystem, dessen Frequenzen direkt die momentanen
Energien widerspiegeln.


8. Stereopositionierung über Winkel
Der Winkel eines Agenten im Raum bestimmt sein Stereo-Panning:
• sinusbasiertes Mapping
• kontinuierliche Links-Rechts-Verteilung
Damit wird die räumliche Struktur des 3D-Feldes hörbar gemacht.

9. Synthese der Eigenzeit-Audioausgabe
Die akustische Rekonstruktion erfolgt wie folgt:
1. Für jedes Audiosample wird anhand der physikalischen Eigenzeit der passende
Simulationsindex bestimmt.
2. Für jeden Agenten wird die Frequenz für diesen Zeitpunkt abgerufen.
3. 256 Sinusoszillatoren werden gleichzeitig berechnet.
4. Die Phasen werden fortlaufend akkumuliert, um echte Oszillation zu erzeugen.
5. Alle Oszillatoren werden stereo gemischt.
6. Das Signal wird normalisiert und als WAV gespeichert.
Das resultierende Audio ist daher eine direkte akustische Projektion des Energieflusses der
Simulation, nicht eine künstliche Nachvertonung.

10. Ergebnisdaten
Die Simulation erzeugt:
• Eigenzeitreihe
• vollständige Energietrajektorie der 256 Agenten
• Audioausgabe (Eigenzeit → Klang)
• optional auswertbare Diagramme für Phasenraum, Spektrum, RMS usw.

Zusammenfassung in einem Satz
Das Skript implementiert ein dreidimensionales energiegekoppeltes Agentensystem mit
stochastischer Dynamik, selbstorganisierter Zeitentwicklung und direkter akustischer Projektion
des Energieflusses, wodurch die emergente Struktur des MCM-Feldes hör- und analysierbar
wird.

---

Autor: Pascal.E
