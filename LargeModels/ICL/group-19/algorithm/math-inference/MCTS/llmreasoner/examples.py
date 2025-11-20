
def get_llm_reasoner_examples():
    examples = {}
    examples["format_prompt"] = " Given a question, please decompose it into sub-questions. For each sub-question, please answer it in a complete sentence, the ending answer need to be \"\\boxed{}\". When the original question is answerable, please start the subquestion with \"Now we can answer the final question: \"."
    examples["format_question"] = "Question {idx}: {question}"
    examples["format_subquestion"] = "Question {idx}.{sub_idx}:"
    examples["format_subanswer"] = "Answer {idx}.{sub_idx}:"
    examples["format_last_question"] = "Now we can answer the final question: "
    examples["last_question_query"] = "we can answer the final question"
    examples["format_examples"] = [
    {
        "Question 1:": "The center of the circle with equation $x^2+y^2=8x-6y-20$ is the point $(x,y)$. What is $x+y$?",
        "Question 1.1:": "What is the standard form of the equation of a circle?",
        "Answer 1.1:": "The standard form of the equation of a circle is $(x - a)^2 + (y - b)^2 = r^2$, where $(a, b)$ is the center of the circle and $r$ is the radius, so the standard form is \\boxed{(x - a)^2 + (y - b)^2 = r^2}.",
        "Question 1.2:": "How can we rewrite the given equation $x^2 + y^2 = 8x - 6y - 20$ in a form suitable for completing the square?",
        "Answer 1.2:": "We first rearrange the equation to group like terms: $x^2 - 8x + y^2 + 6y = -20$, so the suitable form is \\boxed{x^2 - 8x + y^2 + 6y = -20}.",
        "Question 1.3:": "How do we complete the square for the expression $x^2 - 8x + y^2 + 6y = -20$?",
        "Answer 1.3:": "We complete the square: $x^2 - 8x = (x - 4)^2 - 16$, and $y^2 + 6y = (y + 3)^2 - 9$. Substituting back gives $(x - 4)^2 - 16 + (y + 3)^2 - 9 = -20$, which simplifies to \\boxed{(x - 4)^2 + (y + 3)^2 = 5}.",
        "Question 1.4:": "What is the center of the circle based on the standard form $(x - 4)^2 + (y + 3)^2 = 5$?",
        "Answer 1.4:": "The center of the circle is the point \\boxed{(4, -3)}.",
        "Question 1.5:": "Now we can answer the final question: What is $x + y$?",
        "Answer 1.5:": "Given the center is $(4, -3)$, we compute $x + y = 4 + (-3) = \\boxed{1}."
    },
    {
        "Question 2:": "If three standard 6-faced dice are rolled, what is the probability that the sum of the face up integers is 16?",
        "Question 2.1:": "What is the total number of possible outcomes when rolling three 6-faced dice?",
        "Answer 2.1:": "Each die has 6 faces, and since the rolls are independent, the total number of outcomes is $6 \\times 6 \\times 6 = \\boxed{216}$.",
        "Question 2.2:": "What are the possible combinations of three dice that sum to 16?",
        "Answer 2.2:": "We list all ordered triples $(a, b, c)$ such that $1 \\leq a, b, c \\leq 6$ and $a + b + c = 16$. These combinations are: \n\n(6,6,4), (6,4,6), (4,6,6), (6,5,5), (5,6,5), (5,5,6). \n\nThus, there are \\boxed{6} favorable outcomes.",
        "Question 2.3:": "Now we can answer the final question: What is the probability that the sum is 16?",
        "Answer 2.3:": "There are 6 favorable outcomes out of 216 total outcomes, so the probability is \\( \\frac{6}{216} = \\frac{1}{36} \\), hence \\boxed{\\frac{1}{36}}."
    },
    {
        "Question 3:": "Piravena must make a trip from $A$ to $B$, then from $B$ to $C$, then from $C$ to $A$.  Each of these three parts of the trip is made entirely by bus or entirely by airplane.  The cities form a right-angled triangle as shown, with $C$ a distance of 3000 km from $A$ and with $B$ a distance of 3250 km from $A$. To take a bus, it costs Piravena $\\$0.15$ per kilometer. To take an airplane, it costs her a $\\$100$ booking fee, plus $\\$0.10$ per kilometer. [asy]\n\npair A, B, C;\n\nC=(0,0);\n\nB=(0,1250);\n\nA=(3000,0);\n\ndraw(A--B--C--A);\n\nlabel(\"A\", A, SE);\n\nlabel(\"B\", B, NW);\n\nlabel(\"C\", C, SW);\n\nlabel(\"3000 km\", (A+C)/2, S);\n\nlabel(\"3250 km\", (A+B)/2, NE);\n\ndraw((0,125)--(125,125)--(125,0));\n\n[/asy]\n\nPiravena chose the least expensive way to travel between cities.  What was the total cost?",
        "Question 3.1:": "What is the distance from $B$ to $C$ using the Pythagorean Theorem?",
        "Answer 3.1:": "Since triangle $ABC$ is a right triangle with the right angle at $C$, we use the Pythagorean Theorem: $AB^2 = AC^2 + BC^2$. We are given $AB = 3250$ km and $AC = 3000$ km, so:\n\n$BC^2 = AB^2 - AC^2 = 3250^2 - 3000^2 = 10562500 - 9000000 = 1562500$, so $BC = \\sqrt{1562500} = \\boxed{1250}$ km.",
        "Question 3.2:": "What is the cost of traveling each leg by bus and by airplane?",
        "Answer 3.2:": "For bus: $\\$0.15$ per km. For airplane: $\\$100$ booking fee plus $\\$0.10$ per km. \n- $AB = 3250$ km → Bus: $0.15 \\times 3250 = \\$487.50$, Airplane: $100 + 0.10 \\times 3250 = \\$425.00$.\n- $BC = 1250$ km → Bus: $0.15 \\times 1250 = \\$187.50$, Airplane: $100 + 0.10 \\times 1250 = \\$225.00$.\n- $CA = 3000$ km → Bus: $0.15 \\times 3000 = \\$450.00$, Airplane: $100 + 0.10 \\times 3000 = \\$400.00$.",
        "Question 3.3:": "What is the cheapest option for each leg of the trip?",
        "Answer 3.3:": "- For $AB$, airplane is cheaper: \\boxed{\\$425.00}.\n- For $BC$, bus is cheaper: \\boxed{\\$187.50}.\n- For $CA$, airplane is cheaper: \\boxed{\\$400.00}.",
        "Question 3.4:": "Now we can answer the final question: What is the total cost?",
        "Answer 3.4:": "The total cost is $425.00 + 187.50 + 400.00 = \\boxed{\\$1012.50}."
    },
    {
        "Question 4:": "Find the minimum value of \\[\\sin^4 x + \\frac{3}{2} \\cos^4 x,\\] as $x$ varies over all real numbers.",
        "Question 4.1:": "Can we express the expression in terms of a single trigonometric function?",
        "Answer 4.1:": "Yes, using the identity $\\sin^2 x + \\cos^2 x = 1$, we let $s = \\sin^2 x$, so $\\cos^2 x = 1 - s$. Then $\\sin^4 x = s^2$, and $\\cos^4 x = (1 - s)^2$. So the expression becomes:\n\\[s^2 + \\frac{3}{2}(1 - s)^2\\] \\boxed{s^2 + \\frac{3}{2}(1 - s)^2}",
        "Question 4.2:": "What is the simplified expression in terms of $s$?",
        "Answer 4.2:": "We expand:\n\\[s^2 + \\frac{3}{2}(1 - 2s + s^2) = s^2 + \\frac{3}{2} - 3s + \\frac{3}{2}s^2 = \\frac{5}{2}s^2 - 3s + \\frac{3}{2}\\]\nSo the simplified expression is \\boxed{\\frac{5}{2}s^2 - 3s + \\frac{3}{2}}.",
        "Question 4.3:": "What is the minimum value of \\( \\frac{5}{2}s^2 - 3s + \\frac{3}{2} \\) on the interval $0 \\leq s \\leq 1$?",
        "Answer 4.3:": "This is a quadratic in $s$, opening upwards. The minimum occurs at $s = \\frac{-(-3)}{2 \\cdot \\frac{5}{2}} = \\frac{3}{5}$. Substituting:\n\\[\\frac{5}{2} \\cdot \\left(\\frac{3}{5}\\right)^2 - 3 \\cdot \\frac{3}{5} + \\frac{3}{2} = \\frac{5}{2} \\cdot \\frac{9}{25} - \\frac{9}{5} + \\frac{3}{2} = \\frac{45}{50} - \\frac{90}{50} + \\frac{75}{50} = \\frac{30}{50} = \\boxed{\\frac{3}{5}}\\]",
        "Question 4.4:": "Now we can answer the final question: What is the minimum value of \\(\\sin^4 x + \\frac{3}{2} \\cos^4 x\\)?",
        "Answer 4.4:": "The minimum value of the expression is \\boxed{\\frac{3}{5}}."
    },
    {
        "Question 5:": "Find the remainder when $2x^6 - x^4 + 4x^2 - 7$ is divided by $x^2 + 4x + 3$.",
        "Question 5.1:": "What is the degree of the divisor $x^2 + 4x + 3$?",
        "Answer 5.1:": "The degree of the divisor $x^2 + 4x + 3$ is \\boxed{2}.",
        "Question 5.2:": "What is the degree of the remainder when dividing by a quadratic polynomial?",
        "Answer 5.2:": "When dividing by a polynomial of degree 2, the remainder will be a polynomial of degree less than 2, i.e., of the form $ax + b$. \\boxed{ax + b}",
        "Question 5.3:": "How do we find the remainder of a polynomial division?",
        "Answer 5.3:": "We can perform polynomial division or use the method of undetermined coefficients: write\n\\[2x^6 - x^4 + 4x^2 - 7 = (x^2 + 4x + 3)Q(x) + ax + b\\]\nand solve for constants $a$ and $b$ by plugging in values of $x$ that make the divisor zero. \\boxed{Use undetermined coefficients with roots of the divisor}",
        "Question 5.4:": "What are the roots of the divisor $x^2 + 4x + 3$?",
        "Answer 5.4:": "Factoring gives $x^2 + 4x + 3 = (x + 1)(x + 3)$, so the roots are \\boxed{-1} and \\boxed{-3}.",
        "Question 5.5:": "What is the value of the original polynomial at $x = -1$ and $x = -3$?",
        "Answer 5.5:": "Let $P(x) = 2x^6 - x^4 + 4x^2 - 7$.\n\n$P(-1) = 2(-1)^6 - (-1)^4 + 4(-1)^2 - 7 = 2 - 1 + 4 - 7 = -2$.\n\n$P(-3) = 2(-3)^6 - (-3)^4 + 4(-3)^2 - 7 = 2(729) - 81 + 4(9) - 7 = 1458 - 81 + 36 - 7 = 1406$.\nSo \\boxed{P(-1) = -2}, \\boxed{P(-3) = 1406}.",
        "Question 5.6:": "How do we determine $a$ and $b$ in the remainder $ax + b$?",
        "Answer 5.6:": "We know $P(x) = (x^2 + 4x + 3)Q(x) + ax + b$, so plugging in $x = -1$ and $x = -3$:\n\n$P(-1) = a(-1) + b = -a + b = -2$\n$P(-3) = a(-3) + b = -3a + b = 1406$",
        "Question 5.7:": "Solve the system of equations to find $a$ and $b$.",
        "Answer 5.7:": "From $-a + b = -2$ and $-3a + b = 1406$, subtracting the first from the second:\n\n$(-3a + b) - (-a + b) = 1406 - (-2) \\Rightarrow -2a = 1408 \\Rightarrow a = -704$\nSubstitute back: $-(-704) + b = -2 \\Rightarrow 704 + b = -2 \\Rightarrow b = -706$.\nSo the remainder is \\boxed{-704x - 706}.",
        "Question 5.8:": "Now we can answer the final question: What is the remainder when $2x^6 - x^4 + 4x^2 - 7$ is divided by $x^2 + 4x + 3$?",
        "Answer 5.8:": "The remainder is \\boxed{-704x - 706}."
    },
    ]
    examples["useful_prompt"] = "Given a question and some sub-questions, determine whether the last sub-question is useful to answer the question. Output 'Yes' or 'No', and a reason.\n\n"
    examples["useful_prefix"] = "Is Useful {idx}.{sub_idx}:"
    examples["useful_examples"] = [
    {
        "Question 1:": "What value of x will give the maximum value for -x^2 - 6x + 12?",
        "Question 1.1:": "What are the coefficients a and b in the quadratic expression -x^2 - 6x + 12?",
        "Question 1.2:": "Using the vertex formula x = -b / (2a), what is the x-coordinate of the parabola's vertex?",
        "Question 1.3:": "What is the maximum (y-)value of -x^2 - 6x + 12 at that vertex?",
        "Is Useful 1.3:": "No. The actual maximum value of the quadratic is not required to determine which x gives that maximum."
    },
    {
        "Question 2:": "What is the total number of digits used when the first 2002 positive even integers are written?",
        "Question 2.1:": "How many of the first 2002 positive even integers have exactly one digit?",
        "Question 2.2:": "How many of the first 2002 positive even integers have exactly two digits?",
        "Question 2.3:": "How many of the first 2002 positive even integers have exactly three digits?",
        "Question 2.4:": "How many of the first 2002 positive even integers have exactly four digits?",
        "Is Useful 2.4:": "Yes. Knowing the count of four-digit numbers is necessary to determine their digit contribution and, together with the earlier sub-questions, to compute the overall total."
    },
    {
        "Question 3:": "Wallace's water heater is twice the size of Catherine's water heater. If the capacity of Wallace's water heater is 40 gallons and it's 3/4 full, calculate the total number of gallons of water they both have if Catherine's water heater is also full with water to 3/4 of its capacity.",
        "Question 3.1:": "How much water is in Wallace's water heater?",
        "Question 3.2:": "How much water do they have in total?",
        "Is Useful 3.2:": "No. It is too hard to answer the new question based on the current information."
    },
    {
        "Question 1:": "Find the point in the plane 3x - 4y + 5z = 30 that is closest to the point (1, 2, 3).",
        "Question 1.1:": "What is a normal vector to the plane 3x - 4y + 5z = 30?",
        "Question 1.2:": "What is the vector equation of the line passing through (1, 2, 3) in the direction of this normal vector?",
        "Question 1.3:": "For what value of the parameter t does this line intersect the plane 3x - 4y + 5z = 30?",
        "Is Useful 1.3:": "Yes. Determining t locates the intersection point, which is exactly the point on the plane closest to (1, 2, 3)."
    },
    ]
    return examples

