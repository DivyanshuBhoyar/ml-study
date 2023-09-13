def momentum(theta_w, theta_b, v_w, v_b, beta, alpha):
    v_w = beta * v_w + (1 - beta) * theta_w
    v_b = beta * v_b + (1 - beta) * theta_b

    return v_w, v_b
