# Diplomarbeit
Durch die hier hochgeladenen Skripte, Daten und Modelle können die Arbeitsprozesse, der zugehörigen Diplomarbeit, nachempfunden, sowie dessen Ergebnisse repliziert werden.
## Skripte:
Die Dateien mit dem Präfix *Chatbot* beziehen sich auf das erste Szenario zur Pattern-Reconstruction-Attack, entsprechend beziehen sich Dateien mit dem Präfix *Email* auf das zweiten Szenario zur Keyword-Inference-Attack. \
Der Algorithmus, der im Chatbot-Szenario für die Erstellung der Ausweisnummern angewandt wurde, ist im Skript *AusweisNr* zu finden. Die Anwendung de Algorithmus und die darauffolgende Erstellung der Label ist im Skript *DatenNr* festgehalten. \
Das Einfügen er Schlüsselwörter und Namen geschah durch das *Keywords* Skript.\
Die Erstellung der GPLM-Vektoren, die für Trainings- sowie Testdaten verwendet worden sind, ist mittels der im *gplm*-Ordner spezifizierten Funktionen durchgeführt worden. 
Alle Modelle, die für die Attacken oder die weiterführenden Analysen im Anhang, verwendet wurden befinden sich im Ordner *models*. Das Training dieser Modelle geschah durch die Skripte mit dem Suffix *Train*.\
Die Daten zu den in der Arbeit verwendeten Diagrammen und Tabellen lassen sich durch die Skripte mit dem Suffix *Test* oder *Anhang* generieren.\
Schlussendlich wird in den *AttackFunction* Skripten der komplette Ablauf der beiden durchgeführten Angriffe automatisiert durchgegangen. Der Input beider Funktion entspricht einen GPLM-Vektor und der Output jeweils den Ergebnissen der Attacken. Demnach ist der Output für eine erfolgreiche Attacke im ersten Szenario die vollständig rekonstruierte Ausweisnummer und für das zweite die gefundenen verdächtigen Wörter sowie Mitarbeiternamen. Um den zweiten Angriff in einem realen Kontext nützlicher zu gestalten werden hier die Wahrscheinlichkeiten, mit denen ein Schlüsselwort in der jeweiligen Nachricht vorkommt, ebenfalls angegeben. 
## Daten
Die zugehörigen Daten sind unter folgendem Link verfügbar: \
https://drive.google.com/file/d/1yVvv7GZcL4kyw8HpAhtQV3cFk-SBWDGq/view?usp=sharing \
Die Daten sollten in den Ordner *Diplomarbeit* entpackt werden.\
Für das erste Szenario sind jeweils die zum Training verwendeten Nachrichten und Ausweisnummern in zwei getrennten Datensätzen (*Messages*, *AusweisNr*) sowie der für die Tests verwendete Datensatz mit 19.00 Nachrichten und 1.000 Nummern (*Test*) hochgeladen.\
Da der Enron-Emails-Datensatz sehr umfangreich ist, wurde hier nur eine Vorauswahl der Daten hochgeladen, die entsprechend zum Trainieren und Testen der Keyword-Inference-Attack (*WB_Train*, *Test*) und des SpamFilters (*SpamFilter_Train*, *SpamFilter_Test*) verwendet worden sind. \
Um das Einfügen der Schlüsselwörter exemplarisch durchführen zu können wurde der Yelp-Reviews-Datensatz einmal in der Rohform (*yelp.csv*) und einmal nach dem Einfügen (*BB_Train*) hochgeladen und eine Ausschnitt aus den dem Enron-Datensatz (*Enron*)

