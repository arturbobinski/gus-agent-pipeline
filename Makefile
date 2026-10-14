PYTHON := .venv/bin/python
PIP := .venv/bin/pip
RUN_MODE := $(if $(filter 1,$(LIVE)),--live,--dry-run)

.PHONY: setup week board render reroll review assemble post reply analyze test

setup:
	python3.12 -m venv .venv
	$(PIP) install -r requirements.txt

week:
	$(PYTHON) -m gus.write $(RUN_MODE)

board:
	$(PYTHON) -m gus.board $(RUN_MODE) $(if $(EP),--episode $(EP),)

render:
	$(PYTHON) -m gus.render $(RUN_MODE) $(if $(EP),--episode $(EP),) $(if $(filter 1,$(FORCE)),--force,)

reroll:
	$(PYTHON) -m gus.render $(RUN_MODE) --reroll --episode $(EP) --clip $(CLIP) $(if $(PROMPT),--prompt "$(PROMPT)",) $(if $(FRAME),--composition-frame "$(FRAME)",) $(if $(filter 1,$(FORCE)),--force,)

review:
	$(PYTHON) -m gus.review $(RUN_MODE)

assemble:
	$(PYTHON) -m gus.assemble

post:
	$(PYTHON) -m gus.post $(RUN_MODE)

reply:
	$(PYTHON) -m gus.reply $(RUN_MODE)

analyze:
	$(PYTHON) -m gus.analyze $(RUN_MODE)

test:
	$(PYTHON) -m pytest
