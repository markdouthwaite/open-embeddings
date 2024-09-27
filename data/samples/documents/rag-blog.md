If you’ve been on or near the Generative AI (Gen AI) rollercoaster that has characterised the last year or so, it is likely you will have come across the acronym ‘RAG’. RAG stands for ‘Retrieval-Augmented Generation’. It is an approach commonly credited to a 2021 paper by a team led by researchers from Facebook AI Research (FAIR). This approach enables language models to generate ‘more specific, diverse and factual language’ than previously possible. Practically, this means that Large Language Model (LLM) driven systems using RAG are typically:

Less susceptible to hallucinations (e.g. factual errors)
Easily updated/extended to include new information (without retraining the LLM)
Able to provide references for their own factual statements
Additionally, the team claim that human evaluators involved in their research subjectively preferred RAG-powered LLM outputs over those without RAG. It is no surprise that with the explosion in Large Language Models (LLMs) like GPT-4, Claude and PaLM 2 in the last year, this method has rapidly become a standard tool in the LLM development toolbox.

This article aims to provide you with an accessible introduction to RAG. It’s aimed at getting ML practitioners up to speed with the basics, but if you’re semi-technical or simply curious about the world of Gen AI, you may find it useful too.

The basic idea
LLMs make mistakes. Factual errors in responses to user queries can arise for all sorts of reasons, but a common contributor to these mistakes is simply that the training data used to create them lacks relevant or accurate information about the given query. For example, the query may pertain to proprietary information that was not available to the LLM, or the query pertains to events that occurred after the model was trained. In the absence of this data, it is common for LLMs to hallucinate reasonable-sounding but inaccurate responses, or to return out-of-date information. Clearly, for many use cases this is not acceptable.

Technically, the information stored within the model (i.e. a representation of the data it was trained on) is sometimes referred to as ‘parametric memory’. Updating this parametric memory with new data can be expensive: retraining (or fine tuning) can be costly in developer time and compute costs.

Another strategy is to use ‘non-parametric’ memory (i.e. information stored outside of the model) to augment the LLM’s responses. In RAG, this works by combining non-parametric memory in the form of an external search-like capability (the ‘retriever’), the results of which are used to ‘augment’ the parametric memory of an LLM (the ‘generator’) by providing additional ‘context’ in the LLM’s inputs.

Concretely, this process involves using a user's prompt to find documents or document snippets relevant to that prompt (this is the job of the retriever). This document store acts as the model's non-parametric memory. The user's prompt and the retrieved document or document snippets are next combined to produce an augmented prompt which is then used to generate an output from an LLM. Here's a diagram showing how that fits together:


Figure 1: A simple diagram showing how RAG works.
For those interested in where these approaches came from, while this approach is now commonly referred to as RAG, the idea of combining parametric and non-parametric memory in a similar manner predates the commonly cited RAG paper itself (check out REALM and some earlier work on Open-Domain Question and Answering, for example). The original paper is more focussed on the specifics of how to create and train flexible, high-quality retriever and generator models rather than the general pattern itself.

A no-code worked example…
Time to bring this all alive with a small worked example. You can follow along at home with this using GPT-4 in the Open AI API Playground, if you like. Remember though: LLMs produce non-deterministic responses - your responses will differ to those shown.

Here’s a short input prompt asking GPT-4 about the recently completed (as of October 2023) OSIRIS-REx mission from NASA:

What is the current status of NASA's OSIRIS-REx mission?
And here is GPT-4’s response:

NASA's OSIRIS-REx spacecraft is returning to Earth after successfully collecting 
samples from asteroid Bennu. The spacecraft departed Bennu on May 10, 2021, and 
it is expected to deliver the samples to Earth on September 24, 2023. Even 
though OSIRIS-REx has left the asteroid, mission scientists will continue to 
study the data collected during the spacecraft's two and a half years in orbit 
around Bennu.
In this case, the response is out of date. As of writing in October 2023, the sample has already been returned and OSIRIS-REx has been repurposed for a new mission. Not ideal if you're relying on this for something important!

As you may have guessed, that is where RAG comes in. You can improve the quality of the response by injecting additional context into the prompt. Ordinarily, this step would be performed automatically by the retriever. In this case, the retriever is simply a flawed human being that has manually-sourced some information from Wikipedia about the OSIRIS-REx mission. You'll have to use your imagination. Here's an example of the document snippets the 'retriever' might retrieve from its data store in this situation:

1. OSIRIS-REx was a NASA asteroid-study and sample-return mission that visited and 
collected samples from 101955 Bennu, a carbonaceous near-Earth asteroid. 
2. Following the completion of the primary mission, the spacecraft is planned to 
conduct a flyby of asteroid 99942 Apophis as OSIRIS-APEX.
3. The OSIRIS-REx team prepared the spacecraft for the next phase of the mission, 
the return cruise to Earth. On 7 April 2021, it completed its final flyover of 
Bennu and began slowly drifting away from the asteroid. On 10 May 2021, the 
spacecraft departed the Bennu vicinity and began its two-year journey to Earth 
with the asteroid sample.
4. On 24 September 2023, at 4:42 a.m. MDT (UTC-06:00), at a distance of 63,000 
miles (101,000 kilometers) from Earth, it ejected the sample return capsule.
5. On 11 October 2023, the recovered capsule was opened to reveal a "first look" 
at the asteroid sample contents.
The next step is to combine the retrieved documents or document snippets with the input prompt. Here’s what the augmented prompt could look like:

Here are some facts about the OSIRIS-REx mission:

1. OSIRIS-REx was a NASA asteroid-study and sample-return mission that visited and 
collected samples from 101955 Bennu, a carbonaceous near-Earth asteroid. 
2. Following the completion of the primary mission, the spacecraft is planned to 
conduct a flyby of asteroid 99942 Apophis as OSIRIS-APEX.
3. The OSIRIS-REx team prepared the spacecraft for the next phase of the mission, 
the return cruise to Earth. On 7 April 2021, it completed its final flyover of 
Bennu and began slowly drifting away from the asteroid. On 10 May 2021, the 
spacecraft departed the Bennu vicinity and began its two-year journey to Earth 
with the asteroid sample.
4. On 24 September 2023, at 4:42 a.m. MDT (UTC-06:00), at a distance of 63,000 
miles (101,000 kilometers) from Earth, it ejected the sample return capsule.
5. On 11 October 2023, the recovered capsule was opened to reveal a "first look" 
at the asteroid sample contents.

Today is 29 October 2023.

What is the current status of NASA's OSIRIS-REx mission?
While these five facts were taken manually from the OSIRIS-REx Wikipedia page, in a fully automated setup, the retriever would rely on some form of external document database (in this case a database of all Wikipedia pages) to return relevant snippets of documents and inject them into the prompt fully automatically.

There’s some nuance to how these documents are typically processed, stored and accessed for this use-case. If you're interested in seeing a little more of this aspect, check out the interactive notebook linked later in this post.

Notice also a small addition to the augmented prompt in the form of today's date. Dates can be confusing for GPT-4: it was trained many months ago and can sometimes make reasoning mistakes if you are not specific about these sorts of things.

With this updated prompt, the response from GPT-4 looks like this:

The OSIRIS-REx mission has successfully collected and returned a sample from the 
asteroid 101955 Bennu to Earth. The spacecraft departed Bennu vicinity on 
May 10, 2021, and made a two-year journey back to Earth. The sample return capsule 
was ejected from the spacecraft on September 24, 2023, and it landed on Earth. On 
October 11, 2023, the recovered capsule was opened to reveal a "first look" at the 
asteroid sample contents. Plans are underway for the spacecraft to conduct a flyby 
of asteroid 99942 Apophis as OSIRIS-APEX.
This is factually accurate and up to date - exactly the sort of response you’re likely to want. Pretty good, right? That is the minimal case of RAG in a nutshell.

So far so good. Another useful aspect of RAG is that you can get it to directly reference the source documents it has used. You can achieve this with a little more prompt engineering. Few-shot learning can help coax an LLM to produce an output in the desired format. Here’s some additional context you could insert into the start of your previous prompt to teach the LLM how to reference RAG-derived information in its responses:

Your response must include references to any numbered facts provided, for 
example, given the fact:

1. The Apollo Lunar Module (Eagle) landed on the moon on July 20, 1969 at 20:17 
UTC.

A valid response format to the question "When did man land on the moon?" would be:

Apollo 11 landed on the moon on July 20, 1969 [1]
GPT-4 then gives this response:

The OSIRIS-REx mission has completed its primary task of visiting and collecting 
samples from 101955 Bennu, a carbonaceous near-Earth asteroid [1]. After slowly 
drifting away from Bennu, the spacecraft began its two-year journey to Earth with 
the asteroid sample on 10 May 2021 [3]. The sample return capsule was ejected on 
24 September 2023 at a distance of 63,000 miles from Earth [4] and was opened on 
11 October 2023 to reveal a "first look" at the asteroid sample contents [5]. 
The spacecraft is now planned to conduct a flyby of asteroid 99942 Apophis, which 
is named as OSIRIS-APEX [2].
Importantly, while this example uses numerical references for simplicity, the same technique works for adding hyperlinks to text and/or adding footnotes to the LLM's response. In other words, it relatively straightforward to get an LLM to produce an output with interactive references! Here's how it could look with a little prompt engineering:


Figure 2: A sample of a nicely rendered RAG-powered LLM output with linked references.
Putting it into practice with Python
If you’re a technical person, you may now be curious about how this can all be brought together programmatically. Time for some code. If you’d like to follow along with this section, you can clone this GitHub repository, install the provided requirements and boot up the notebook with Python 3.10 or greater.

GitHub - markdouthwaite/llm-explorations: A repository containing interactive examples of different LLM use-cases.
A repository containing interactive examples of different LLM use-cases. - GitHub - markdouthwaite/llm-explorations: A repository containing interactive examples of different LLM use-cases.

GitHub
markdouthwaite

Alternatively, you can open it in Google Colab and follow along in your browser right now. Enjoy!

Google Colaboratory


Closing thoughts
As you can see, at its heart, RAG is a super simple, yet very powerful idea. It allows almost anyone to quickly integrate any data source — be that a dump of Wikipedia articles, recent news stories, video transcripts or proprietary knowledge bases (KBs) — as non-parametric memory for an LLM. In practice, this produces higher quality, more accurate responses from the LLM augmented in this way. To summarise, benefits of the approach include:

Flexibility - when you have a basic RAG setup in place, it is incredibly easy to add or update this non-parametric memory for your LLM of choice.
Scalability - using common vector database technologies you can query many millions of documents in a fraction of a second, and ultimately
Quality - LLMs that use RAG typically hallucinate less and consistently return more accurate responses to queries.
As with any technique, however, it is no silver bullet. While it is typically straightforward to implement, the addition of RAG for your LLM does increase the complexity of your LLM-powered application. As is often the case, the difference between a toy RAG application and a production application can be significant - particularly if you have stringent performance requirements. Additionally, the performance of RAG is only as good as the documents in your non-parametric memory. As always, if they are poor-quality or inaccurate, your LLM's responses will likely follow suite.

However, the success of RAG in driving up the quality of LLM responses is impressive. It has contributed significantly to the boom in vector database technologies over the last year, and its popularity is such that it is rare to find a modern LLM-powered application without some flavour of RAG under the hood. It is a powerful tool in your toolbox.