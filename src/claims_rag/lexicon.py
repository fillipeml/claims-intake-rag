"""The words the corpus cannot teach itself.

`WordSegmenter.from_texts` learns its vocabulary from the lines that kept their spaces, which
is elegant and, measured, insufficient. The words it needs most are the ones it never sees:
``atendimento``, ``afastamento``, ``habituais`` and ``necessidade`` all came back with a
frequency of zero, because every line containing them is a line the recognition model merged.
A corpus cannot teach itself a word it only ever shows welded to its neighbours.

So the vocabulary is seeded. Not with a general Portuguese dictionary — with the words a motor
claims file is made of, which is a few hundred of them and is the sort of glossary any claims
operation already maintains. Corpus frequencies are added on top, so a word that *is* observed
gets a proper probability and the seeded entries act as a floor.

The split is deliberate and worth keeping: `segmentation.py` holds the algorithm and knows
nothing about insurance; this file holds the domain and knows nothing about dynamic
programming. Changing domain means replacing this file.
"""

from __future__ import annotations

from collections import Counter

#: Function words, auxiliaries and the connective tissue of Brazilian administrative prose.
_GENERAL = """
a o e as os de da do das dos em no na nos nas um uma uns umas por para com que se ao aos
pelo pela pelos pelas este esta estes estas esse essa isso aquele aquela aquilo seu sua
seus suas foi era ser sido sendo tem ter teve tinha havia ha sao esta estao houve nao sim
mais menos muito pouco todo toda todos todas outro outra outros outras mesmo mesma entre
sobre sob ate desde apos antes durante conforme segundo quando onde como porque tambem
ja ainda apenas somente bem mal maior menor melhor pior primeiro segundo terceiro
necessidade possibilidade responsabilidade validade quantidade gravidade
um dois tres quatro cinco seis sete oito nove dez quinze vinte trinta sessenta noventa
"""

#: The vocabulary of a Brazilian motor claim: the documents, the parties, the money, the
#: vehicle, the injury and the process.
_DOMAIN = """
aviso sinistro apolice segurado segurada seguradora corretor beneficiario terceiro
numero data local natureza evento descricao declaracao documento documentos complementares
regulacao indenizacao cobertura coberturas contratada contratadas limite franquia premio
vigencia exclusao exclusoes clausula condicoes gerais particulares especiais
boletim ocorrencia delegacia circunscricao autoridade policial relato condutor conducao
habilitacao carteira categoria valida vencida vitima vitimas fatal lesao lesoes
veiculo veiculos automovel placa chassi renavam marca modelo ano fabricacao cor
colisao capotamento incendio roubo furto alagamento enchente granizo abalroamento
traseira dianteira lateral direita esquerda frontal impacto choque danos avarias
orcamento reparo reparos oficina credenciada funilaria pintura mecanica peca pecas
mao obra servico servicos substituicao troca quantidade unitario subtotal total geral
parachoque paralama capo porta vidro parabrisa farol lanterna retrovisor airbag
suspensao motor cambio radiador condensador chicote modulo
laudo medico medica paciente atendimento atendido atendida consciente orientado orientada
exame inicial neurologico neurologicas agudas alteracoes avaliacao diagnostico prognostico
afastamento atividades habituais dias repouso retorno ambulatorial reavaliacao clinica
fratura contusao escoriacao trauma internacao alta hospitalar
pagamento quitacao recibo deposito transferencia parcela parcelas vencimento
analise aprovacao recusa pendencia solicitacao prazo resposta protocolo registro
rodovia via marginal preferencial cruzamento semaforo semaforica intermitente sinalizacao
alca acesso canteiro pista faixa chuva aderencia perda guincho remocao
"""


def base_vocabulary() -> Counter[str]:
    """The seed counts.

    Every seeded word gets the same small count, so the lexicon says *this is a word* without
    asserting how common it is. Corpus observations then do the ranking, which is the right
    division: the glossary knows what exists, the corpus knows what is frequent here.
    """
    counts: Counter[str] = Counter()
    for source in (_GENERAL, _DOMAIN):
        for word in source.split():
            counts[word.lower()] += 3
    return counts


#: Exposed so a test can assert the two halves stay separate and neither silently empties.
GENERAL_WORDS = frozenset(w.lower() for w in _GENERAL.split())
DOMAIN_WORDS = frozenset(w.lower() for w in _DOMAIN.split())
