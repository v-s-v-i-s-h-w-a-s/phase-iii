import React, { useState, useEffect, useRef } from 'react';
import { Stage, Layer, Rect, Circle, Line, Text, Group } from 'react-konva';
import axios from 'axios';
import io from 'socket.io-client';
import {
  Container,
  Paper,
  TextField,
  Button,
  Typography,
  Box,
  Card,
  CardContent,
  Grid,
  Alert,
  LinearProgress,
  Chip,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  List,
  ListItem,
  ListItemText,
  Divider
} from '@mui/material';
import { PlayArrow, Stop, Settings, Analytics } from '@mui/icons-material';

// Basketball court dimensions
const COURT_WIDTH = 940;
const COURT_HEIGHT = 500;

// Player colors by position
const POSITION_COLORS = {
  'PG': '#FF6B6B',  // Red
  'SG': '#4ECDC4',  // Teal
  'SF': '#45B7D1',  // Blue
  'PF': '#96CEB4',  // Green
  'C': '#FFEAA7'    // Yellow
};

function BasketballCourt({ players, onPlayerMove, isSimulating }) {
  const stageRef = useRef();

  const courtElements = () => {
    const elements = [];

    // Court background
    elements.push(
      <Rect
        key="court-bg"
        width={COURT_WIDTH}
        height={COURT_HEIGHT}
        fill="#8B4513"
        stroke="#FFFFFF"
        strokeWidth={3}
      />
    );

    // Three-point arc
    elements.push(
      <Line
        key="three-point"
        points={[
          100, 150,  // Left corner
          237, 50,   // Left arc
          470, 25,   // Top
          703, 50,   // Right arc  
          840, 150   // Right corner
        ]}
        stroke="#FFFFFF"
        strokeWidth={2}
        closed={false}
      />
    );

    // Free throw lane
    elements.push(
      <Rect
        key="ft-lane"
        x={390}
        y={50}
        width={160}
        height={190}
        fill="transparent"
        stroke="#FFFFFF"
        strokeWidth={2}
      />
    );

    // Free throw circle
    elements.push(
      <Circle
        key="ft-circle"
        x={470}
        y={180}
        radius={60}
        fill="transparent"
        stroke="#FFFFFF"
        strokeWidth={2}
      />
    );

    // Center circle
    elements.push(
      <Circle
        key="center-circle"
        x={470}
        y={250}
        radius={60}
        fill="transparent"
        stroke="#FFFFFF"
        strokeWidth={2}
      />
    );

    // Basket
    elements.push(
      <Circle
        key="basket"
        x={470}
        y={50}
        radius={8}
        fill="#FF6B35"
        stroke="#000000"
        strokeWidth={1}
      />
    );

    return elements;
  };

  const playerElements = () => {
    return players.map((player, index) => (
      <Group
        key={`player-${index}`}
        x={player.position[0]}
        y={player.position[1]}
        draggable={!isSimulating}
        onDragEnd={(e) => {
          if (onPlayerMove) {
            onPlayerMove(index, e.target.x(), e.target.y());
          }
        }}
      >
        <Circle
          radius={15}
          fill={POSITION_COLORS[player.position] || '#999999'}
          stroke="#000000"
          strokeWidth={2}
          shadowColor="black"
          shadowBlur={5}
          shadowOffset={{ x: 2, y: 2 }}
          shadowOpacity={0.3}
        />
        <Text
          text={player.position}
          fontSize={10}
          fontStyle="bold"
          fill="#FFFFFF"
          offsetX={8}
          offsetY={5}
        />
        {player.has_ball && (
          <Circle
            radius={20}
            stroke="#FF6B35"
            strokeWidth={3}
            fill="transparent"
          />
        )}
      </Group>
    ));
  };

  return (
    <Paper elevation={3} style={{ padding: '10px', marginBottom: '20px' }}>
      <Stage
        width={COURT_WIDTH}
        height={COURT_HEIGHT}
        ref={stageRef}
        style={{ border: '2px solid #ccc' }}
      >
        <Layer>
          {courtElements()}
          {playerElements()}
        </Layer>
      </Stage>
    </Paper>
  );
}

function PlayCreator() {
  const [playDescription, setPlayDescription] = useState('');
  const [formation, setFormation] = useState('5-out');
  const [players, setPlayers] = useState([]);
  const [parsedPlay, setParsedPlay] = useState(null);
  const [simulationResult, setSimulationResult] = useState(null);
  const [isSimulating, setIsSimulating] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [socket, setSocket] = useState(null);

  // Initialize socket connection
  useEffect(() => {
    const newSocket = io('http://localhost:5000');
    setSocket(newSocket);

    newSocket.on('simulation_update', (data) => {
      console.log('Simulation update:', data);
      // Update players positions during simulation
      if (data.players) {
        setPlayers(data.players);
      }
    });

    return () => newSocket.close();
  }, []);

  // Initialize default players
  useEffect(() => {
    initializeFormation(formation);
  }, [formation]);

  const initializeFormation = (formationType) => {
    const formations = {
      '5-out': [
        { id: 'player_1', position: 'PG', team: 'offense', position: [470, 400], has_ball: true },
        { id: 'player_2', position: 'SG', team: 'offense', position: [600, 350], has_ball: false },
        { id: 'player_3', position: 'SF', team: 'offense', position: [340, 350], has_ball: false },
        { id: 'player_4', position: 'PF', team: 'offense', position: [600, 250], has_ball: false },
        { id: 'player_5', position: 'C', team: 'offense', position: [340, 250], has_ball: false }
      ],
      '4-out-1-in': [
        { id: 'player_1', position: 'PG', team: 'offense', position: [470, 400], has_ball: true },
        { id: 'player_2', position: 'SG', team: 'offense', position: [600, 350], has_ball: false },
        { id: 'player_3', position: 'SF', team: 'offense', position: [340, 350], has_ball: false },
        { id: 'player_4', position: 'PF', team: 'offense', position: [570, 300], has_ball: false },
        { id: 'player_5', position: 'C', team: 'offense', position: [470, 200], has_ball: false }
      ]
    };

    setPlayers(formations[formationType] || formations['5-out']);
  };

  const handlePlayerMove = (playerIndex, newX, newY) => {
    const updatedPlayers = [...players];
    updatedPlayers[playerIndex].position = [newX, newY];
    setPlayers(updatedPlayers);
  };

  const parsePlay = async () => {
    if (!playDescription.trim()) {
      setError('Please enter a play description');
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      const response = await axios.post('/api/parse_play', {
        description: playDescription,
        formation: formation
      });

      setParsedPlay(response.data);
      setError('');

      // Update players based on parsed play
      if (response.data.players) {
        setPlayers(response.data.players);
      }
    } catch (err) {
      setError('Failed to parse play: ' + (err.response?.data?.error || err.message));
    }

    setIsLoading(false);
  };

  const simulatePlay = async () => {
    if (!parsedPlay) {
      setError('Please parse a play first');
      return;
    }

    setIsSimulating(true);
    setError('');

    try {
      const response = await axios.post('/api/simulate_play', {
        players: players,
        actions: parsedPlay.actions,
        duration: parsedPlay.duration
      });

      setSimulationResult(response.data);

      // Start real-time simulation updates
      if (socket) {
        socket.emit('start_simulation', {
          players: players,
          actions: parsedPlay.actions
        });
      }

      // Simulate for the play duration
      setTimeout(() => {
        setIsSimulating(false);
      }, parsedPlay.duration * 1000);

    } catch (err) {
      setError('Failed to simulate play: ' + (err.response?.data?.error || err.message));
      setIsSimulating(false);
    }
  };

  const optimizePlay = async () => {
    if (!simulationResult) {
      setError('Please simulate a play first');
      return;
    }

    try {
      const response = await axios.post('/api/optimize_play', {
        simulation_result: simulationResult,
        original_description: playDescription
      });

      // Update parsed play with optimized version
      setParsedPlay(response.data.optimized_play);
      setPlayDescription(response.data.optimized_description);

    } catch (err) {
      setError('Failed to optimize play: ' + (err.response?.data?.error || err.message));
    }
  };

  return (
    <Container maxWidth="xl">
      <Typography variant="h3" component="h1" gutterBottom align="center" sx={{ mt: 2, mb: 4 }}>
        Basketball Play Creator
      </Typography>

      <Grid container spacing={3}>
        {/* Left Panel - Controls */}
        <Grid item xs={12} lg={4}>
          <Paper elevation={3} sx={{ p: 3, mb: 2 }}>
            <Typography variant="h5" gutterBottom>
              Play Design
            </Typography>

            <TextField
              fullWidth
              multiline
              rows={4}
              label="Play Description"
              placeholder="Enter your play description (e.g., 'PG passes to SG, center sets screen, cut to basket')"
              value={playDescription}
              onChange={(e) => setPlayDescription(e.target.value)}
              sx={{ mb: 2 }}
            />

            <FormControl fullWidth sx={{ mb: 2 }}>
              <InputLabel>Formation</InputLabel>
              <Select
                value={formation}
                label="Formation"
                onChange={(e) => setFormation(e.target.value)}
              >
                <MenuItem value="5-out">5-Out</MenuItem>
                <MenuItem value="4-out-1-in">4-Out-1-In</MenuItem>
                <MenuItem value="1-4-high">1-4-High</MenuItem>
              </Select>
            </FormControl>

            <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
              <Button
                variant="contained"
                onClick={parsePlay}
                disabled={isLoading || isSimulating}
                sx={{ flex: 1 }}
              >
                Parse Play
              </Button>

              <Button
                variant="contained"
                color="success"
                onClick={simulatePlay}
                disabled={!parsedPlay || isSimulating}
                startIcon={isSimulating ? <Stop /> : <PlayArrow />}
                sx={{ flex: 1 }}
              >
                {isSimulating ? 'Simulating...' : 'Simulate'}
              </Button>
            </Box>

            <Button
              variant="outlined"
              onClick={optimizePlay}
              disabled={!simulationResult}
              startIcon={<Analytics />}
              fullWidth
            >
              Optimize Play
            </Button>

            {(isLoading || isSimulating) && (
              <LinearProgress sx={{ mt: 2 }} />
            )}

            {error && (
              <Alert severity="error" sx={{ mt: 2 }}>
                {error}
              </Alert>
            )}
          </Paper>

          {/* Parsed Play Info */}
          {parsedPlay && (
            <Paper elevation={3} sx={{ p: 3, mb: 2 }}>
              <Typography variant="h6" gutterBottom>
                Parsed Play: {parsedPlay.name}
              </Typography>

              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                Confidence: {Math.round(parsedPlay.confidence * 100)}%
              </Typography>

              <Typography variant="subtitle2" gutterBottom>
                Actions:
              </Typography>
              <List dense>
                {parsedPlay.actions.map((action, index) => (
                  <ListItem key={index} sx={{ py: 0 }}>
                    <ListItemText
                      primary={`${action.type}: ${action.description}`}
                      secondary={`Player: ${action.player} | Time: ${action.timestamp.toFixed(1)}s`}
                    />
                  </ListItem>
                ))}
              </List>
            </Paper>
          )}
        </Grid>

        {/* Center Panel - Court */}
        <Grid item xs={12} lg={5}>
          <BasketballCourt
            players={players}
            onPlayerMove={handlePlayerMove}
            isSimulating={isSimulating}
          />

          {/* Player Legend */}
          <Paper elevation={3} sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Players
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              {Object.entries(POSITION_COLORS).map(([position, color]) => (
                <Chip
                  key={position}
                  label={position}
                  sx={{ backgroundColor: color, color: '#fff' }}
                  size="small"
                />
              ))}
            </Box>
          </Paper>
        </Grid>

        {/* Right Panel - Analysis */}
        <Grid item xs={12} lg={3}>
          {simulationResult && (
            <Paper elevation={3} sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Simulation Results
              </Typography>

              <Card variant="outlined" sx={{ mb: 2 }}>
                <CardContent>
                  <Typography color="text.secondary" gutterBottom>
                    Success Probability
                  </Typography>
                  <Typography variant="h4" component="div">
                    {Math.round(simulationResult.success_probability * 100)}%
                  </Typography>
                </CardContent>
              </Card>

              <Card variant="outlined" sx={{ mb: 2 }}>
                <CardContent>
                  <Typography color="text.secondary" gutterBottom>
                    Score Probability
                  </Typography>
                  <Typography variant="h4" component="div">
                    {Math.round(simulationResult.final_score_probability * 100)}%
                  </Typography>
                </CardContent>
              </Card>

              <Typography variant="subtitle2" gutterBottom>
                Key Interactions:
              </Typography>
              <List dense>
                {simulationResult.key_interactions.map((interaction, index) => (
                  <ListItem key={index} sx={{ py: 0 }}>
                    <ListItemText primary={interaction} />
                  </ListItem>
                ))}
              </List>

              <Divider sx={{ my: 2 }} />

              <Typography variant="subtitle2" gutterBottom>
                Tactical Analysis:
              </Typography>
              {Object.entries(simulationResult.tactical_analysis).map(([key, value]) => (
                <Box key={key} sx={{ mb: 1 }}>
                  <Typography variant="body2">
                    {key.replace('_', ' ').toUpperCase()}: {Math.round(value * 100)}%
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={value * 100}
                    sx={{ height: 6, borderRadius: 3 }}
                  />
                </Box>
              ))}

              <Divider sx={{ my: 2 }} />

              <Typography variant="subtitle2" gutterBottom>
                Optimization Suggestions:
              </Typography>
              <List dense>
                {simulationResult.optimization_suggestions.map((suggestion, index) => (
                  <ListItem key={index} sx={{ py: 0 }}>
                    <ListItemText
                      primary={suggestion}
                      primaryTypographyProps={{ variant: 'body2' }}
                    />
                  </ListItem>
                ))}
              </List>
            </Paper>
          )}
        </Grid>
      </Grid>
    </Container>
  );
}

export default PlayCreator;
