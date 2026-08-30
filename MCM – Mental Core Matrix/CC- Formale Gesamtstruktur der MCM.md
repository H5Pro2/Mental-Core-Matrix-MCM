# CC- Formale Gesamtstruktur der MCM
## Dokumentstatus

- **Status:** Hypothetische Modellgrundlage
- **Funktion:** Definition oder Formalisierung von Begriffen und Beziehungen innerhalb der MCM
- **Geltungsbereich:** Der Text beschreibt einen Modellrahmen und keine gesicherte Erklärung der Wirklichkeit.
- **Prüfung:** Formale oder technische Tests beziehen sich auf die jeweilige Umsetzung, nicht automatisch auf die MCM insgesamt.



Dokument CC
Formale Gesamtstruktur der MCM

1. Grundraum
Im Rahmen dieses hypothetischen Modells wird die Mental Core Matrix (MCM) auf einem
eindimensionalen Zustandsraum definiert als
𝑋 = [−3, +3] ⊂ 𝑅
mit
• 𝑥 < 0: negative Spannungsseite
• 𝑥 = 0: Zentrum
• 𝑥 > 0: positive Spannungsseite
Dieser Raum bildet den gemeinsamen Kernraum der reinen und der psychologischen MCM.

2. Reine MCM: kontinuierliche Feldform
Die reine MCM wird als kontinuierliche Zustandsdichte
𝜌(𝑥, 𝑡) ≥ 0, 𝑥 ∈ [−3, +3], 𝑡 ≥ 0
mit Normierung
∫
+3
−3
𝜌(𝑥, 𝑡)𝑑𝑥 = 1

modelliert.
Die Dynamik des Feldes wird durch eine Drift-Diffusions-Gleichung beschrieben:
𝜕𝜌(𝑥, 𝑡)
 𝜕𝑡 = 𝜕
𝜕𝑥 (𝑣(𝑥)𝑝(𝑥, 𝑡)) + 𝐷 𝜕2𝜌(𝑥, 𝑡)
𝜕𝑥²
mit
𝑣(𝑥) = −𝑘𝑥, 𝑘 > 0, 𝐷 ≥ 0
Hierbei beschreibt
• 𝑣(𝑥): Rückführung zum Zentrum
• 𝐷: Fluktuation / Varianz
Diese Form entspricht der reinen energetischen Grunddynamik der MCM.

3. Potenzialform


Äquivalent kann die Rückführung über ein Potenzial beschrieben werden:
𝑉(𝑥) = 1
2 𝑎𝑥2, 𝑎 > 0
mit
𝑣(𝑥) = − 𝑑𝑉
𝑑𝑥 = −𝑎𝑥
Das Zentrum (x=0) kann damit als Attraktor des Modells interpretiert werden.

4. Psychologische MCM: Zustandsform eines Einzelzustands
Die psychologische MCM beschreibt einen einzelnen mentalen Zustand als zeitabhängige
Trajektorie
𝑥(𝑡) ∈ [−3, +3]
mit Dynamik

𝑑𝑥
𝑑𝑡 = −𝑘𝑥 + 𝐼(𝑡) + 𝜂(𝑡)
wobei
• −𝑘𝑥: Rückführung zum Zentrum
• 𝐼(𝑡): externer Einfluss / Reiz
• 𝜂(𝑡): stochastische Varianz
bezeichnet.

5. Spannungsfunktion
Die Spannung eines Zustands wird definiert als Distanz zum Zentrum:
𝑆(𝑥) =∣ 𝑥 ∣
optional nichtlinear:
𝑆(𝑥) =∣ 𝑥 ∣ 𝛼, 𝛼 > 1
Damit können extreme Zustände strukturell stärker gewichtet werden.

6. Zonenstruktur
Die psychologische MCM verwendet eine Intervallzerlegung des Kernraums:
𝐺1 = [−3, −1], 𝐺2 = [−1, +1], 𝐺3 = [+1, +2], 𝐺4 = [+2, +3]
mit Zentrum


𝑍 = {0}
sowie Übergangsbereichen
𝑈1 = [−1.5, −0.5], 𝑈2 = [−0.2, +0.2], 𝑈3 = [+0.2, +0.8], 𝑈4 = [+1.8, +2.4]
Diese Bereiche sind diskrete Interpretationszonen auf einem kontinuierlichen Raum.

7. Archetypenabbildung
Die psychologische Ebene wird formal als interpretative Abbildung auf dem Kernraum
modelliert:
𝛷: 𝑋 → 𝐴
wobei
• 𝒳 = [−3, +3] der energetische Kernraum ist
• 𝐴 die Menge psychologischer Archetypen und Übergangszonen bezeichnet
Die psychologische MCM ist damit keine eigene Feldstruktur, sondern eine symbolische
Projektion auf denselben Grundraum.
Das ist die formale Brücke zwischen AA und BB.

8. Feldgrößen
Für die reine MCM werden globale Größen definiert als

⟨𝑥⟩(𝑡) = ∫
+3
−3
𝑥𝜌(𝑥, 𝑡)𝑑𝑥
und
𝑉𝑎𝑟(𝑥)(𝑡) = ∫
+3
−3
(𝑥 − ⟨𝑥⟩)2𝜌(𝑥, 𝑡)𝑑𝑥
Diese Größen können als Maß für globale Auslenkung und Feldunruhe interpretiert werden.

9. Zeitfunktion
Innerhalb des Modells kann eine effektive Zeitdynamik an Feldaktivität oder Spannung gekoppelt
werden.
Für die reine MCM:

𝑇𝑒𝑓𝑓(𝑡) = 𝑐 ⋅ 𝑉𝑎𝑟(𝑥)(𝑡)
Für die psychologische MCM:


𝑇𝑒𝑟𝑙𝑒𝑏𝑡(𝑥) = 𝑐 ⋅∣ 𝑥 ∣
mit 𝑐 > 0.

10. Gesamtform der MCM
Die Gesamtstruktur der MCM kann damit formal als zweischichtiges Modell beschrieben
werden:

𝑀𝐶𝑀 = (𝑋, 𝜌, 𝑥(𝑡), 𝛷)
mit
• 𝑋: gemeinsamer Kernraum
• 𝜌(𝑥, 𝑡): reine energetische Feldform
• 𝑥(𝑡): psychologische Zustandsdynamik
• 𝛷: symbolische Abbildung in Archetypen / psychologische Zonen

---

Autor: Pascal.E
